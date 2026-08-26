#!/usr/bin/env python3
"""
Give Claude Code real reasoning-effort control over a local llama-server.

Claude Code sends its effort setting as Anthropic's `output_config: {"effort": ...}`.
llama-server's Anthropic-format /v1/messages endpoint ignores that field, and it also
ignores a top-level `reasoning_effort` (which only works on /v1/chat/completions), so
the model always runs at whatever `--chat-template-kwargs` the server was started with.

This proxy sits between the two and rewrites

    "output_config": {"effort": "high"}
        -> "chat_template_kwargs": {"reasoning_effort": "high"}

which IS honoured on /v1/messages. The Qwen3.8 template aliases high -> xhigh itself,
so the value passes through unchanged; anything the template would reject is dropped
rather than sent (an unknown effort makes the template raise and the request 500).

Usage:
    ./claude-code-effort-proxy.py &
    ANTHROPIC_BASE_URL=http://127.0.0.1:8899 \
    ANTHROPIC_AUTH_TOKEN=local \
    ANTHROPIC_MODEL=qwen3.8-27b \
    claude

Environment:
    QWEN_UPSTREAM   upstream llama-server        (default http://nas-server.fritz.box:8000)
    QWEN_PORT       port to listen on            (default 8899)
    QWEN_BIND       address to bind              (default 127.0.0.1)
    QWEN_FORCE      ignore Claude Code's effort, always send this one (low|medium|xhigh)
    QWEN_VERBOSE    set to 1 to log every rewrite
"""

import http.client
import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = os.environ.get("QWEN_UPSTREAM", "http://nas-server.fritz.box:8000").rstrip("/")
PORT = int(os.environ.get("QWEN_PORT", "8899"))
BIND = os.environ.get("QWEN_BIND", "127.0.0.1")
FORCE = os.environ.get("QWEN_FORCE") or None
VERBOSE = os.environ.get("QWEN_VERBOSE") == "1"

# The three values the Qwen3.8 template accepts. It aliases 'high' -> 'xhigh' internally,
# so 'high' is safe to pass through; every other value raises a Jinja exception.
ACCEPTED = {"low", "medium", "high", "xhigh"}

# Headers we must not copy verbatim in either direction.
HOP_BY_HOP = {
    "host", "content-length", "accept-encoding", "connection",
    "keep-alive", "transfer-encoding", "te", "trailer", "upgrade", "proxy-authorization",
}


def log(*a):
    if VERBOSE:
        print(*a, file=sys.stderr, flush=True)


def rewrite(body: bytes) -> bytes:
    """Move output_config.effort into chat_template_kwargs.reasoning_effort."""
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return body  # not JSON we understand - pass it through untouched
    if not isinstance(payload, dict):
        return body

    effort = FORCE
    if effort is None:
        oc = payload.get("output_config")
        if isinstance(oc, dict):
            effort = oc.get("effort")
    if not isinstance(effort, str) or effort not in ACCEPTED:
        if effort is not None:
            log(f"[effort] dropping unsupported effort {effort!r}")
        return body

    ctk = payload.get("chat_template_kwargs")
    if not isinstance(ctk, dict):
        ctk = {}
    if "reasoning_effort" in ctk:
        return body  # caller was explicit - don't override it
    ctk["reasoning_effort"] = effort
    payload["chat_template_kwargs"] = ctk
    log(f"[effort] output_config.effort={effort} -> chat_template_kwargs.reasoning_effort={effort}")
    return json.dumps(payload).encode()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "qwen-effort-proxy"

    def log_message(self, *a):
        pass  # keep the terminal clean; use QWEN_VERBOSE for the interesting bits

    def _relay(self, method: str):
        body = b""
        length = self.headers.get("content-length")
        if length:
            body = self.rfile.read(int(length))
        if method == "POST" and self.path.split("?")[0].endswith("/v1/messages"):
            body = rewrite(body)

        req = urllib.request.Request(UPSTREAM + self.path, data=body or None, method=method)
        for k, v in self.headers.items():
            if k.lower() not in HOP_BY_HOP:
                req.add_header(k, v)
        req.add_header("Accept-Encoding", "identity")  # no gzip: we stream bytes through

        try:
            upstream = urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            upstream = e  # error responses still carry a body worth forwarding
        except OSError as e:
            msg = json.dumps({"type": "error", "error": {
                "type": "api_error",
                "message": f"effort proxy could not reach {UPSTREAM}: {e}"}}).encode()
            self.send_response(502)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(msg)))
            self.end_headers()
            self.wfile.write(msg)
            return

        with upstream:
            declared = upstream.headers.get("content-length")
            self.send_response(upstream.status)
            for k, v in upstream.headers.items():
                if k.lower() not in HOP_BY_HOP:
                    self.send_header(k, v)
            if declared is not None:
                self.send_header("content-length", declared)
            else:
                self.send_header("transfer-encoding", "chunked")  # SSE lands here
            self.end_headers()

            try:
                while True:
                    # read1() hands over whatever has arrived instead of waiting for a
                    # full buffer - without it, streamed tokens would arrive in clumps.
                    chunk = upstream.read1(65536)
                    if not chunk:
                        break
                    if declared is not None:
                        self.wfile.write(chunk)
                    else:
                        self.wfile.write(f"{len(chunk):X}\r\n".encode() + chunk + b"\r\n")
                    self.wfile.flush()
                if declared is None:
                    self.wfile.write(b"0\r\n\r\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass  # client hung up mid-stream (Ctrl-C in Claude Code); nothing to do

    def do_POST(self):
        self._relay("POST")

    def do_GET(self):
        self._relay("GET")


def main():
    server = ThreadingHTTPServer((BIND, PORT), Handler)
    server.daemon_threads = True
    print(f"qwen effort proxy: http://{BIND}:{PORT} -> {UPSTREAM}"
          + (f" (forcing effort={FORCE})" if FORCE else ""), file=sys.stderr, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
