#!/usr/bin/env python3
"""
Moderna (MRNA) Dashboard — stock price + news in one page.

Run:
    python3 moderna_dashboard.py          # -> http://localhost:8000
    python3 moderna_dashboard.py 9000     # custom port

Standard library only. Data sources (no API key needed):
  * Yahoo Finance chart API -> price, chart, day range, 52-week range
  * Google News RSS         -> general Moderna news (last 14 days)
  * Yahoo Finance RSS       -> MRNA market headlines
Responses are cached server-side for 5 minutes.
"""
import json
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Lock

CACHE_TTL = 300  # seconds
# NOTE: Yahoo Finance returns 429 for the old Linux Chrome/124 UA string
# (widely used by bots); this newer Windows UA passes their checks.
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
      "Accept": "application/json"}
VALID_RANGES = {"1d", "5d", "1mo", "3mo", "6mo", "1y"}
VALID_INTERVALS = {"5m", "15m", "1h", "1d"}

_cache = {}
_lock = Lock()


def http_get(url: str, timeout: float = 15.0) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def cached(key, fetcher):
    with _lock:
        entry = _cache.get(key)
        if entry and time.time() - entry[0] < CACHE_TTL:
            return entry[1]
    result = fetcher()
    with _lock:
        _cache[key] = (time.time(), result)
    return result


# ---------------------------------------------------------------- prices

def fetch_price(rng, interval):
    """Returns (dict, None) on success, (None, error) on failure."""
    last_err = "Yahoo Finance request failed"
    result = None
    for host in ("query1", "query2"):
        url = (f"https://{host}.finance.yahoo.com/v8/finance/chart/MRNA"
               f"?range={rng}&interval={interval}")
        try:
            payload = json.loads(http_get(url))
            result = (payload.get("chart") or {}).get("result")
            if result:
                break
        except Exception as exc:
            last_err = str(exc)
            result = None
    if not result:
        return None, last_err

    d = result[0]
    meta = d.get("meta", {})
    timestamps = d.get("timestamp") or []
    quote = ((d.get("indicators") or {}).get("quote") or [{}])[0]
    closes = quote.get("close") or []
    highs = quote.get("high") or []
    lows = quote.get("low") or []
    vols = quote.get("volume") or []

    points = []
    for i, t in enumerate(timestamps):
        if i >= len(closes) or closes[i] is None:
            continue
        points.append({
            "t": t,
            "c": round(closes[i], 2),
            "h": round(highs[i], 2) if i < len(highs) and highs[i] is not None else None,
            "l": round(lows[i], 2) if i < len(lows) and lows[i] is not None else None,
            "v": vols[i] if i < len(vols) and vols[i] is not None else 0,
        })

    data = {
        "symbol": meta.get("symbol", "MRNA"),
        "currency": meta.get("currency", "USD"),
        "price": meta.get("regularMarketPrice"),
        "prevClose": meta.get("chartPreviousClose") or meta.get("previousClose"),
        "open": meta.get("regularMarketOpen"),
        "dayHigh": meta.get("regularMarketDayHigh"),
        "dayLow": meta.get("regularMarketDayLow"),
        "volume": meta.get("regularMarketVolume"),
        "w52High": meta.get("fiftyTwoWeekHigh"),
        "w52Low": meta.get("fiftyTwoWeekLow"),
        "marketState": meta.get("marketState", ""),
        "points": points,
    }
    return data, None


# ------------------------------------------------------------------ news

def parse_rss(data, default_source):
    root = ET.fromstring(data)
    items = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        link = (item.findtext("link") or "").strip()
        pub = (item.findtext("pubDate") or "").strip()
        src_el = item.find("source")
        source = (src_el.text or "").strip() if src_el is not None and src_el.text else default_source
        if source and title.endswith(" - " + source):
            title = title[: -(len(source) + 3)]
        ts = 0.0
        if pub:
            try:
                ts = parsedate_to_datetime(pub).timestamp()
            except Exception:
                ts = 0.0
        if title and link:
            items.append({"title": title, "link": link, "source": source, "ts": ts})
    return items


def gnews_url(query):
    return ("https://news.google.com/rss/search?hl=en-US&gl=US&ceid=US:en&q="
            + urllib.parse.quote(query))


def fetch_news():
    sources = [
        (gnews_url("Moderna when:14d"), "Google News"),
        (gnews_url("Moderna stock when:7d"), "Google News"),
        ("https://feeds.finance.yahoo.com/rss/2.0/headline?s=MRNA&region=US&lang=en-US",
         "Yahoo Finance"),
    ]
    items, errors = [], []
    for url, default_src in sources:
        try:
            items.extend(parse_rss(http_get(url), default_src))
        except Exception as exc:
            errors.append(str(exc))
    if not items:
        msg = "; ".join(errors) if errors else "No items returned"
        return None, "All news sources failed: " + msg

    seen, unique = set(), []
    for it in sorted(items, key=lambda x: x["ts"], reverse=True):
        key = re.sub(r"[^a-z0-9]+", "", it["title"].lower())[:80]
        if not key or key in seen:
            continue
        seen.add(key)
        unique.append(it)
    return unique[:40], None


# ------------------------------------------------------------- http server

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        try:
            if url.path == "/":
                self._send(INDEX_HTML.encode(), "text/html; charset=utf-8")
            elif url.path == "/api/price":
                q = urllib.parse.parse_qs(url.query)
                rng = (q.get("range") or ["3mo"])[0]
                interval = (q.get("interval") or ["1d"])[0]
                if rng not in VALID_RANGES or interval not in VALID_INTERVALS:
                    self._json({"error": "invalid range/interval"}, 400)
                    return
                data, err = cached(("price", rng, interval),
                                   lambda: fetch_price(rng, interval))
                if err:
                    self._json({"error": err}, 502)
                else:
                    self._json(data)
            elif url.path == "/api/news":
                data, err = cached(("news",), fetch_news)
                if err:
                    self._json({"error": err}, 502)
                else:
                    self._json(data)
            else:
                self._json({"error": "not found"}, 404)
        except Exception as exc:
            try:
                self._json({"error": str(exc)}, 500)
            except Exception:
                pass


# ----------------------------------------------------------------- frontend

INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Moderna (MRNA) Dashboard</title>
<link rel="icon" href="data:image/svg+xml,%3Csvg%20xmlns='http://www.w3.org/2000/svg'%20viewBox='0%200%2064%2064'%3E%3Crect%20width='64'%20height='64'%20rx='14'%20fill='%23ff5c8a'/%3E%3Ctext%20x='32'%20y='43'%20font-size='34'%20font-family='Arial'%20font-weight='700'%20text-anchor='middle'%20fill='%23fff'%3EM%3C/text%3E%3C/svg%3E">
<style>
:root{--bg:#0b0e14;--card:#121826;--border:#1f2937;--text:#e8ecf4;--muted:#8b94a7;
--accent:#ff5c8a;--up:#2fd180;--down:#ff5470;--radius:14px}
*{box-sizing:border-box}
body{margin:0;background:radial-gradient(1200px 500px at 80% -10%,rgba(255,92,138,.08),transparent),var(--bg);
color:var(--text);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,Arial,sans-serif}
.wrap{max-width:1240px;margin:0 auto;padding:22px;display:flex;flex-direction:column;gap:16px}
header{display:flex;align-items:center;gap:18px;flex-wrap:wrap;background:var(--card);
border:1px solid var(--border);border-radius:var(--radius);padding:18px 22px}
.logo{width:48px;height:48px;border-radius:12px;background:linear-gradient(135deg,#ff5c8a,#ff8f6b);
display:flex;align-items:center;justify-content:center;font-size:24px;font-weight:800;color:#fff}
h1{margin:0;font-size:20px}
.sub{color:var(--muted);font-size:13px;display:flex;gap:8px;align-items:center}
.pill{font-size:11px;padding:2px 9px;border-radius:99px;border:1px solid var(--border);color:var(--muted)}
.pill.live{color:var(--up);border-color:rgba(47,209,128,.4)}
.pricebox{margin-left:auto;text-align:right}
#bigPrice{font-size:34px;font-weight:700;letter-spacing:-.5px}
.badge{display:inline-block;margin-top:2px;font-size:13px;font-weight:600;padding:3px 10px;border-radius:8px}
.badge.up{color:var(--up);background:rgba(47,209,128,.12)}
.badge.down{color:var(--down);background:rgba(255,84,112,.12)}
.controls{display:flex;flex-direction:column;gap:10px;align-items:flex-end}
.ranges{display:flex;background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:3px}
.ranges button{background:none;border:0;color:var(--muted);font:inherit;font-size:13px;font-weight:600;
padding:5px 12px;border-radius:8px;cursor:pointer}
.ranges button.active{background:var(--accent);color:#fff}
.ctlrow{display:flex;align-items:center;gap:10px}
.updated{font-size:12px;color:var(--muted)}
button.ghost{background:none;border:1px solid var(--border);color:var(--text);font:inherit;font-size:13px;
padding:6px 14px;border-radius:10px;cursor:pointer}
button.ghost:hover{border-color:var(--accent)}
button.ghost.small{padding:3px 9px;font-size:12px}
.spin .ic{display:inline-block;animation:spin .8s linear infinite}
@keyframes spin{to{transform:rotate(360deg)}}
main{display:grid;grid-template-columns:1.7fr 1fr;gap:16px;align-items:start}
@media(max-width:980px){main{grid-template-columns:1fr}}
.card{background:var(--card);border:1px solid var(--border);border-radius:var(--radius);padding:18px}
.card-head{display:flex;align-items:center;gap:10px;margin-bottom:12px;font-weight:600;font-size:15px}
.card-head .spacer{margin-left:auto}
.muted{color:var(--muted);font-size:12px}
.chip{font-size:12px;font-weight:700;padding:2px 9px;border-radius:99px}
.chip.up{color:var(--up);background:rgba(47,209,128,.12)}
.chip.down{color:var(--down);background:rgba(255,84,112,.12)}
#chartWrap{position:relative}
canvas{display:block}
.err{display:none;background:rgba(255,84,112,.1);border:1px solid rgba(255,84,112,.35);color:#ffb3c0;
border-radius:10px;padding:8px 12px;font-size:13px;margin-bottom:10px}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:10px;margin-top:14px}
.tile{background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:10px 12px}
.tile.wide{grid-column:1/-1}
.tl{font-size:11px;text-transform:uppercase;letter-spacing:.6px;color:var(--muted);margin-bottom:3px}
.tv{font-size:16px;font-weight:600}
.track{position:relative;height:6px;border-radius:99px;background:linear-gradient(90deg,var(--down),var(--accent),var(--up));
opacity:.9;margin:10px 0 6px}
.dot{position:absolute;top:50%;width:12px;height:12px;border-radius:50%;background:#fff;
border:2px solid var(--bg);transform:translate(-50%,-50%)}
.tracklabels{display:flex;justify-content:space-between;font-size:11px;color:var(--muted)}
#tooltip{position:absolute;pointer-events:none;opacity:0;background:#1b2333;border:1px solid var(--border);
border-radius:10px;padding:8px 12px;font-size:12.5px;transition:opacity .12s;
box-shadow:0 8px 24px rgba(0,0,0,.4);z-index:5;min-width:130px}
.tt-date{color:var(--muted)}
.tt-price{font-size:16px;font-weight:700}
.tt-chg{font-weight:600}
.up{color:var(--up)}.down{color:var(--down)}
#news{list-style:none;margin:0;padding:0;max-height:640px;overflow-y:auto}
#news li{padding:11px 0;border-top:1px solid var(--border)}
#news li:first-child{border-top:0}
#news a{color:var(--text);text-decoration:none;font-size:14px;line-height:1.4}
#news a:hover{color:var(--accent)}
.meta{display:flex;gap:10px;color:var(--muted);font-size:12px;margin-top:3px}
.src{color:var(--accent);opacity:.9}
.empty{color:var(--muted);padding:16px 0}
footer{color:var(--muted);font-size:12px;text-align:center;padding:4px 0 14px}
</style>
</head>
<body>
<div class="wrap">

  <header>
    <div class="logo">M</div>
    <div>
      <h1>Moderna, Inc.</h1>
      <div class="sub"><span>NASDAQ: <span id="sym">MRNA</span></span>
        <span id="mktState" class="pill"></span></div>
    </div>
    <div class="pricebox">
      <div id="bigPrice">—</div>
      <div><span id="change" class="badge up">&nbsp;</span></div>
    </div>
    <div class="controls">
      <div class="ranges">
        <button data-r="1D">1D</button><button data-r="5D">5D</button>
        <button data-r="1M">1M</button><button data-r="3M" class="active">3M</button>
        <button data-r="6M">6M</button><button data-r="1Y">1Y</button>
      </div>
      <div class="ctlrow">
        <button id="refresh" class="ghost"><span class="ic">&#8635;</span> Refresh</button>
        <span id="lastUpdated" class="updated"></span>
      </div>
    </div>
  </header>

  <main>
    <section class="card">
      <div class="card-head">
        <span>Stock Price</span><span id="rangeChg" class="chip up"></span>
        <span class="spacer"></span><span class="muted" id="cur">USD</span>
      </div>
      <div id="priceErr" class="err"></div>
      <div id="chartWrap"><canvas id="chart"></canvas><div id="tooltip"></div></div>
      <div id="stats" class="stats"></div>
    </section>

    <aside class="card">
      <div class="card-head">
        <span>Latest News</span><span class="spacer"></span>
        <button id="refreshNews" class="ghost small"><span class="ic">&#8635;</span></button>
      </div>
      <div id="newsErr" class="err"></div>
      <ul id="news"></ul>
    </aside>
  </main>

  <footer>Data: Yahoo Finance &middot; Google News RSS &middot; cached 5 min server-side &middot; Not financial advice.</footer>
</div>

<script>
const $ = s => document.querySelector(s);
const RANGES = {
  '1D': {range:'1d',  interval:'5m',  label:'Today'},
  '5D': {range:'5d',  interval:'15m', label:'5 Days'},
  '1M': {range:'1mo', interval:'1d',  label:'1 Month'},
  '3M': {range:'3mo', interval:'1d',  label:'3 Months'},
  '6M': {range:'6mo', interval:'1d',  label:'6 Months'},
  '1Y': {range:'1y',  interval:'1d',  label:'1 Year'},
};
let cur = '3M', priceData = null, hoverIdx = -1, geo = null;

const fmtDate = t => {
  const d = new Date(t * 1000);
  return RANGES[cur].interval !== '1d'
    ? d.toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'})
    : d.toLocaleDateString(undefined,{month:'short',day:'numeric',year:'2-digit'});
};
const fmtVol = v => v == null ? '\u2014'
  : v >= 1e9 ? (v/1e9).toFixed(2)+'B'
  : v >= 1e6 ? (v/1e6).toFixed(2)+'M'
  : v >= 1e3 ? (v/1e3).toFixed(1)+'K' : String(v);
const esc = s => String(s).replace(/[&<>"']/g,
  c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const relTime = ts => {
  if (!ts) return '';
  const s = (Date.now() - ts * 1000) / 1000;
  if (s < 60) return 'just now';
  if (s < 3600) return Math.floor(s/60) + 'm ago';
  if (s < 86400) return Math.floor(s/3600) + 'h ago';
  if (s < 7*86400) return Math.floor(s/86400) + 'd ago';
  return new Date(ts*1000).toLocaleDateString();
};

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error('HTTP ' + r.status);
  return r.json();
}
function setErr(id, show, msg) {
  const el = document.getElementById(id);
  el.style.display = show ? 'block' : 'none';
  if (msg) el.textContent = msg;
}

function renderHeader() {
  const d = priceData;
  $('#bigPrice').textContent = d.price != null ? d.price.toFixed(2) : '\u2014';
  if (d.price != null && d.prevClose != null) {
    const diff = d.price - d.prevClose, pct = diff / d.prevClose * 100, up = diff >= 0;
    const chg = $('#change');
    chg.className = 'badge ' + (up ? 'up' : 'down');
    chg.textContent = (up ? '\u25B2 +' : '\u25BC ') + diff.toFixed(2) +
      '  (' + (up?'+':'') + pct.toFixed(2) + '%) vs prev close';
  } else $('#change').innerHTML = '&nbsp;';
  const st = $('#mktState');
  const map = {REGULAR:'Market open', PRE:'Pre-market', POST:'After hours',
               PREPRE:'Pre-market', POSTPOST:'After hours', CLOSED:'Market closed'};
  st.textContent = map[d.marketState] || d.marketState || '';
  st.className = 'pill' + (d.marketState === 'REGULAR' ? ' live' : '');
  $('#cur').textContent = (d.currency || 'USD').toUpperCase();
  $('#lastUpdated').textContent = 'Updated ' + new Date().toLocaleTimeString();
}

function renderStats() {
  const d = priceData;
  const row = (l, v) =>
    `<div class="tile"><div class="tl">${l}</div><div class="tv">${v}</div></div>`;
  let html =
    row('Open', d.open != null ? d.open.toFixed(2) : '\u2014') +
    row('Day High', d.dayHigh != null ? d.dayHigh.toFixed(2) : '\u2014') +
    row('Day Low', d.dayLow != null ? d.dayLow.toFixed(2) : '\u2014') +
    row('Volume', fmtVol(d.volume)) +
    row('52W High', d.w52High != null ? d.w52High.toFixed(2) : '\u2014') +
    row('52W Low', d.w52Low != null ? d.w52Low.toFixed(2) : '\u2014');
  if (d.w52High && d.w52Low && d.price != null) {
    const pos = Math.max(0, Math.min(1,
      (d.price - d.w52Low) / (d.w52High - d.w52Low))) * 100;
    html += `<div class="tile wide"><div class="tl">52-Week Range</div>
      <div class="track"><div class="dot" style="left:${pos.toFixed(1)}%"></div></div>
      <div class="tracklabels"><span>${d.w52Low.toFixed(2)}</span>
      <span>${d.w52High.toFixed(2)}</span></div></div>`;
  }
  $('#stats').innerHTML = html;
  const pts = d.points || [], chip = $('#rangeChg');
  if (pts.length > 1) {
    const pct = (pts[pts.length-1].c / pts[0].c - 1) * 100;
    chip.textContent = (pct >= 0 ? '+' : '') + pct.toFixed(2) + '% over ' + RANGES[cur].label;
    chip.className = 'chip ' + (pct >= 0 ? 'up' : 'down');
  } else chip.textContent = '';
}

function drawChart() {
  const cv = $('#chart'), ctx = cv.getContext('2d');
  const w = cv.parentElement.clientWidth, h = 380;
  const dpr = window.devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = h * dpr;
  cv.style.width = w + 'px'; cv.style.height = h + 'px';
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const pts = (priceData && priceData.points) || [];
  if (!pts.length) return;
  const P = {l:12, r:66, t:16, b:30};
  const plotW = w - P.l - P.r, plotH = h - P.t - P.b;
  let min = Infinity, max = -Infinity;
  pts.forEach(p => { if (p.c < min) min = p.c; if (p.c > max) max = p.c; });
  const pad = (max - min) * 0.08 || 1;
  min -= pad; max += pad;
  const n = pts.length;
  const x = i => P.l + (n === 1 ? plotW/2 : i * plotW / (n - 1));
  const y = v => P.t + (1 - (v - min) / (max - min)) * plotH;
  geo = {P, plotW, plotH, n, w, h, x, y, pts};

  ctx.font = '11px system-ui, sans-serif';
  ctx.textBaseline = 'middle';
  for (let k = 0; k <= 4; k++) {
    const v = min + (max - min) * k / 4, yy = y(v);
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.beginPath(); ctx.moveTo(P.l, yy); ctx.lineTo(w - P.r, yy); ctx.stroke();
    ctx.fillStyle = '#8b94a7'; ctx.textAlign = 'left';
    ctx.fillText('$' + v.toFixed(2), w - P.r + 8, yy);
  }
  ctx.textAlign = 'center'; ctx.textBaseline = 'top';
  const ticks = Math.min(5, n);
  for (let k = 0; k < ticks; k++) {
    const i = Math.round(k * (n - 1) / (ticks - 1 || 1));
    ctx.fillStyle = '#8b94a7';
    ctx.fillText(fmtDate(pts[i].t), x(i), h - P.b + 8);
  }
  let maxV = 0; pts.forEach(p => { if (p.v > maxV) maxV = p.v; });
  if (maxV > 0) {
    const vh = plotH * 0.16, bw = Math.max(1, plotW / n * 0.6);
    pts.forEach((p, i) => {
      const bh = p.v / maxV * vh;
      ctx.fillStyle = 'rgba(255,92,138,0.18)';
      ctx.fillRect(x(i) - bw/2, P.t + plotH - bh, bw, bh);
    });
  }
  const grad = ctx.createLinearGradient(0, P.t, 0, P.t + plotH);
  grad.addColorStop(0, 'rgba(255,92,138,0.30)');
  grad.addColorStop(1, 'rgba(255,92,138,0)');
  ctx.beginPath();
  pts.forEach((p, i) => i === 0 ? ctx.moveTo(x(i), y(p.c)) : ctx.lineTo(x(i), y(p.c)));
  ctx.lineTo(x(n-1), P.t + plotH); ctx.lineTo(x(0), P.t + plotH); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();
  ctx.beginPath();
  pts.forEach((p, i) => i === 0 ? ctx.moveTo(x(i), y(p.c)) : ctx.lineTo(x(i), y(p.c)));
  ctx.strokeStyle = '#ff5c8a'; ctx.lineWidth = 2; ctx.lineJoin = 'round'; ctx.stroke();
  ctx.beginPath(); ctx.arc(x(n-1), y(pts[n-1].c), 3.5, 0, Math.PI * 2);
  ctx.fillStyle = '#ff5c8a'; ctx.fill();
  if (hoverIdx >= 0 && hoverIdx < n) {
    const i = hoverIdx, p = pts[i];
    ctx.strokeStyle = 'rgba(255,255,255,0.25)';
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x(i), P.t); ctx.lineTo(x(i), P.t + plotH); ctx.stroke();
    ctx.setLineDash([]);
    ctx.beginPath(); ctx.arc(x(i), y(p.c), 4.5, 0, Math.PI * 2);
    ctx.fillStyle = '#fff'; ctx.fill();
  }
}

function showTooltip(i) {
  const p = geo.pts[i], tip = $('#tooltip');
  const chg = (p.c / geo.pts[0].c - 1) * 100;
  tip.innerHTML = `<div class="tt-date">${fmtDate(p.t)}</div>
    <div class="tt-price">$${p.c.toFixed(2)}</div>
    <div class="tt-chg ${chg >= 0 ? 'up' : 'down'}">${chg >= 0 ? '+' : ''}${chg.toFixed(2)}% since start</div>`;
  tip.style.opacity = 1;
  let left = geo.x(i) + 14;
  if (left > geo.w - 180) left = geo.x(i) - 178;
  tip.style.left = Math.max(8, left) + 'px';
  tip.style.top = Math.max(6, Math.min(geo.h - 90, geo.y(p.c) - 46)) + 'px';
}

async function loadPrice() {
  try {
    const cfg = RANGES[cur];
    priceData = await getJSON(`/api/price?range=${cfg.range}&interval=${cfg.interval}`);
    renderHeader(); renderStats(); hoverIdx = -1; drawChart();
    setErr('priceErr', false);
  } catch (e) {
    setErr('priceErr', true, 'Could not load price data - check your connection / Yahoo availability.');
  }
}

async function loadNews() {
  try {
    const items = await getJSON('/api/news');
    const el = $('#news');
    if (!items.length) {
      el.innerHTML = '<li class="empty">No news found right now.</li>';
    } else {
      el.innerHTML = items.map(n => `
        <li>
          <a href="${esc(n.link)}" target="_blank" rel="noopener">${esc(n.title)}</a>
          <div class="meta"><span class="src">${esc(n.source || '')}</span>
          <span class="when">${relTime(n.ts)}</span></div>
        </li>`).join('');
    }
    setErr('newsErr', false);
  } catch (e) {
    setErr('newsErr', true, 'Could not load news right now.');
  }
}

const chart = document.getElementById('chart');
chart.addEventListener('mousemove', e => {
  if (!geo) return;
  const rect = chart.getBoundingClientRect();
  let i = Math.round((e.clientX - rect.left - geo.P.l) / geo.plotW * (geo.n - 1));
  hoverIdx = Math.max(0, Math.min(geo.n - 1, i));
  drawChart(); showTooltip(hoverIdx);
});
chart.addEventListener('mouseleave', () => {
  hoverIdx = -1; $('#tooltip').style.opacity = 0; drawChart();
});

document.querySelectorAll('.ranges button').forEach(b => {
  b.addEventListener('click', () => {
    cur = b.dataset.r;
    document.querySelectorAll('.ranges button')
      .forEach(x => x.classList.toggle('active', x === b));
    loadPrice();
  });
});
async function refreshAll() {
  const b = $('#refresh'); b.classList.add('spin');
  await Promise.allSettled([loadPrice(), loadNews()]);
  b.classList.remove('spin');
}
$('#refresh').addEventListener('click', refreshAll);
$('#refreshNews').addEventListener('click', loadNews);
window.addEventListener('resize', () => drawChart());
setInterval(refreshAll, 5 * 60 * 1000);
refreshAll();
</script>
</body>
</html>
"""


def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print()
    print("  Moderna (MRNA) dashboard running at:")
    print(f"  ->  http://localhost:{port}")
    print("  Press Ctrl+C to stop.")
    print()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
