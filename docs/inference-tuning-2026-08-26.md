# Inference tuning session - 2026-08-26

Working notes from an audit of the live Qwen3.8-27B deployment on `nas-server`
(RTX 3090, llama.cpp build `b10577-2c6b141ef`). Covers model choice, the four sampling
parameters, reasoning effort, and the client landscape.

Values marked **measured** were read back from the running server via `GET /props` or probe
requests against `/v1/chat/completions`. Everything else is from the Unsloth docs and the
Qwen3.8 model card.

---

## 1. Model choice for a 24 GB card

`Qwen3.8-27B` (dense) at `UD-Q4_K_XL` remains the right answer. Unsloth's own guidance puts
4-bit Qwen3.8-27B at ~16-19 GB, "comfortably on a 24GB setup".

| Candidate | 4-bit footprint | Verdict |
| :--- | :--- | :--- |
| **Qwen3.8-27B dense** | ~16-19 GB | **Current choice.** Room left for mmproj + MTP head + 192k KV |
| Qwen3.6-35B-A3B (MoE) | ~23 GB | "right at the practical edge" - no room for context or vision |
| Gemma 4 31B | ~17-20 GB | Slight edge on pure vision, loses on agentic reliability |
| Qwen3.8-2.4T-A95B / Max | - | Not single-GPU hardware |

The structural reason is hybrid attention, not parameter count: only 16 of 64 layers are full
attention, so the KV cache grows at ~64 KiB/token instead of ~256 KiB/token. A conventional
dense 27B would exhaust 24 GB somewhere around 48k context.

### On "Dynamic v3"

`UD-` **is** Unsloth Dynamic. The deployed `Qwen3.8-27B-UD-Q4_K_XL.gguf` is already a
Dynamic v3.0 quant - the model card states *"Introducing Dynamic V3.0 GGUFs for SOTA accuracy"*.
There is nothing to switch to. What it buys over stock `Q4_K_M` at the same size:

- **Selective layer upcasting** - important layers keep more bits instead of one uniform bitwidth.
- **>10% better top-1% accuracy** at equal file size vs other providers (Unsloth's benchmark).
- **imatrix calibration tuned for agentic coding and tool-calling**, which matches the
  OpenCode/Hermes workload; a generic wikitext imatrix would not.

Caveat from the docs: Dynamic **1-bit** should not be used for agentic/tool-calling work.
Irrelevant at Q4, relevant if a larger model is ever squeezed in.

---

## 2. The four sampling parameters

They form a pipeline: **top-k -> top-p -> min-p -> temperature** shapes the distribution each
token is drawn from.

| Param | Thinking (default) | Non-thinking | What it does |
| :--- | ---: | ---: | :--- |
| `temperature` | 1.0 | 0.7 | Divides logits pre-softmax. <1 sharpens (repetitive), >1 flattens (incoherent). 1.0 = the model's own distribution untouched |
| `top_p` | 0.95 | 0.80 | Nucleus: keep the smallest set reaching 0.95 cumulative probability. Adaptive to model confidence |
| `top_k` | 20 | 20 | Hard cap on candidates, applied *before* top-p. Blunt tail removal |
| `min_p` | 0.0 (off) | 0.0 (off) | Drop tokens below a fraction of the top token's probability. Off because top-k + top-p already filter; stacking over-constrains |

Non-thinking mode additionally wants `presence_penalty 1.5` to suppress repetition loops.
Leave it at 0.0 in thinking mode - it interferes with the deliberately repetitive structure of
reasoning traces.

**Do not carry `temp 1.0` into agent loops.** For tool-heavy coding, ~0.6 is materially more
reliable.

### The `min_p` trap (measured)

The GGUF carries Qwen's `temp 1.0 / top_p 0.95 / top_k 20` as baked metadata, so those apply
even with no CLI flags. `min_p` is **not** in that metadata and falls back to llama.cpp's own
default of `0.05` - which then stacks on top of top-k/top-p. It must be set explicitly.

Verify what you actually got, never assume:

```bash
curl -s localhost:8000/props | jq '.default_generation_settings.params'
```

---

## 3. Reasoning effort

Independent of sampling. `reasoning_effort` is a **chat-template variable** consumed by Jinja
when the prompt string is built, before the model runs; the samplers act on logits afterwards.
Neither reads the other. They are coupled only by convention (thinking wants temp 1.0,
non-thinking wants 0.6-0.7).

### Accepted values (measured against the deployed template)

The template validates against exactly three values:

- **`xhigh`** - template default. Hard math, tricky debugging, one-shot architecture questions.
- **`medium`** - good interactive default.
- **`low`** - agent loops with many small tool calls, where per-turn latency compounds.
- `high` is silently aliased to `xhigh`.
- **Anything else, including `none`/`off`, raises a Jinja exception and fails with HTTP 500:**

```
500 - Unexpected reasoning effort none. Supported types are xhigh (default), medium, and low.
```

To disable thinking entirely there is a **separate** switch - `reasoning_effort` has no "off":

```json
"chat_template_kwargs": {"enable_thinking": false}
```

Verified to return plain `content` with no `reasoning_content` field. It short-circuits the
effort logic, which is why `reasoning_effort` is never validated on that path.

### Per-request override

`llama-server` accepts `reasoning_effort` as a top-level field and maps it into the template,
so these are equivalent:

```jsonc
{"reasoning_effort": "low"}
{"chat_template_kwargs": {"reasoning_effort": "low"}}
```

Both work on `/v1/chat/completions` **and** `/v1/messages`.

### Measuring which level is actually active

There is no `/props` field for it. Each level injects different instructions into the system
prompt, so prompt length is a reliable fingerprint. Same one-word user message throughout:

| Request | `prompt_tokens` |
| :--- | ---: |
| server default (no kwargs) | 11 |
| explicit `medium` | 11 |
| explicit `low` | 41 |
| explicit `xhigh` | 53 |
| `enable_thinking: false` | 13 |

Note that `medium` injects the *fewest* instruction tokens - it is the untouched baseline,
while `low` and `xhigh` both add explicit steering. So `medium` is cheap twice over: shorter
prompt and shorter thinking traces.

---

## 4. Configuration changes applied

### Server: `/etc/systemd/system/llama-qwen.service`

Backup at `llama-qwen.service.bak-20260826`.

```diff
   --cache-ram 512 \
+  --min-p 0.0 \
+  --chat-template-kwargs '{"reasoning_effort":"medium"}' \
   --mmproj /garagedata/models/gguf/mmproj-Qwen3.8-27B-F16.gguf \
```

```bash
systemd-analyze verify /etc/systemd/system/llama-qwen.service
systemctl daemon-reload && systemctl restart llama-qwen
```

The restart reloads 16 GB of weights and drops the prompt cache - do it between sessions.

**Result (measured):** `min_p` 0.05 -> 0.0; server default effort now indistinguishable from
explicit `medium` (11 prompt tokens) and distinct from `xhigh` (53).

### Client: `~/.config/opencode/config.json`

OpenCode was inheriting the server's `temp 1.0` into agent loops. Per-model `options` is an
arbitrary passthrough object, so anything placed there lands in the request body:

```json
"options": { "temperature": 0.6 }
```

### Client: `~/.qwen/settings.json`

Already correct - pins `temperature 0.6, top_p 0.95, top_k 20` client-side. Note this means
Qwen Code deliberately ignores the server's 1.0, which explains why the WebUI and Qwen Code
give differently-flavored answers to the same prompt. Intended, not drift.

---

## 5. Clients

### Unsloth ships its own

`unsloth start` auto-configures endpoint, key, model and context for a coding agent:

```bash
unsloth start claude \
  --model unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_XL \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --reasoning-effort medium
```

`unsloth run --model ... -p 8888` serves a model directly. There is also Unsloth Desktop.

**Not recommended here:** these replace the llama.cpp stack, losing the MTP draft head, the
vision projector, and the tuned KV quantization this repo is built around.

### Claude Code works as a client - but only with the patched chat template

Build `b10577` exposes a genuine Anthropic-format `/v1/messages` (verified - returns
`type: "message"`, `content: [{type: "thinking", ...}]`, `stop_reason`):

```bash
ANTHROPIC_BASE_URL=http://nas-server.fritz.box:8000 \
ANTHROPIC_AUTH_TOKEN=local \
ANTHROPIC_MODEL=qwen3.8-27b \
CLAUDE_CODE_MAX_CONTEXT_TOKENS=196608 \
claude --settings '{"env":{"CLAUDE_CODE_ATTRIBUTION_HEADER":"0","CLAUDE_CODE_ENABLE_TELEMETRY":"0"}}'
```

Those two env vars are not cosmetic - Unsloth's docs flag ~90% slowdowns against local
endpoints without them.

`CLAUDE_CODE_MAX_CONTEXT_TOKENS` is new here. Claude Code does not recognize `qwen3.8-27b`, so it
assumes a 200k window and sizes auto-compact against it - slightly *larger* than the server's actual
196608. Set it, or auto-compact fires too late.

> **Server-side prerequisite (added 2026-08-26).** Out of the box this does **not** work. The stock
> Qwen3.8 template raises `System message must be at the beginning.` because Claude Code sends its
> agent-type and skill listings as a `role: "system"` message after the first user turn - so `messages`
> roles are `['user', 'system']`. Every request 500s, Claude Code retries 11 times, and the session ends
> with no assistant message (the session title still generates, because that call carries no listings).
> The endpoint itself is fine: streaming, tools, `cache_control`, `tool_choice`, `thinking` and
> `count_tokens` all verified working. Fix is the one-line `--chat-template-file` patch in the README
> ([The chat-template patch](../README.md#the-chat-template-patch)). Verified end to end after the patch:
> the previously-failing 24k-token request returns HTTP 200 and `claude -p` answers normally.

### Changing reasoning effort on the fly

| Client | On the fly? | How |
| :--- | :---: | :--- |
| curl / scripts | yes | `chat_template_kwargs` always; top-level `"reasoning_effort":"low"` **only on `/v1/chat/completions`** |
| llama.cpp WebUI | yes | raw-parameter box in settings |
| OpenCode | per-agent | put `"reasoning_effort"` in the model's `options` passthrough; switching agent switches profile, but not mid-session |
| Claude Code | no | see below |
| Qwen Code | no | no passthrough field; server default only |

**Claude Code's effort setting does not map through** - the status line lies about it.

Claude Code 2.1.246 does not send `reasoning_effort`, and no longer sends a thinking budget
either. Captured from the wire, `effortLevel` maps 1:1 onto Anthropic's `output_config`, while
the thinking block is a fixed `adaptive`:

| Claude Code `effortLevel` | what actually goes on the wire |
| :--- | :--- |
| `low` | `"output_config":{"effort":"low"}`, `"thinking":{"type":"adaptive"}` |
| `medium` | `"output_config":{"effort":"medium"}`, `"thinking":{"type":"adaptive"}` |
| `high` | `"output_config":{"effort":"high"}`, `"thinking":{"type":"adaptive"}` |

`llama-server` drops `output_config` on the floor. Measured on `/v1/messages`, prompt tokens for
an identical request (totals include `cache_read_input_tokens` - ignore that and the prefix cache
will fool you into thinking the prompt shrank):

| request | prompt tokens |
| :--- | ---: |
| baseline (server default `medium`) | 17 |
| `+ "output_config":{"effort":"low"}` | 17 - **ignored** |
| `+ "reasoning_effort":"low"` (top level) | 17 - **ignored** |
| `+ "chat_template_kwargs":{"reasoning_effort":"low"}` | 43 |
| `+ "chat_template_kwargs":{"reasoning_effort":"xhigh"}` | 55 |

So on the Anthropic endpoint **`chat_template_kwargs` is the only per-request lever** - top-level
`reasoning_effort` works on `/v1/chat/completions` (17 -> 43 there) but is dropped on
`/v1/messages`. Claude Code sends neither.

**Net effect:** the banner reads `qwen3.8-27b with high effort` because that is Claude Code's own
`effortLevel` setting, but the model runs at the server default `medium` no matter what you set.
Setting `"effortLevel": "medium"` in `~/.claude/settings.json` makes the banner honest; it changes
nothing about the generation. That is precisely why setting the server default to `medium` matters.

**Fixed 2026-08-26** by [`claude-code-effort-proxy.py`](../claude-code-effort-proxy.py), a client-side
shim that rewrites `output_config.effort` into `chat_template_kwargs.reasoning_effort` on the way
through (the template aliases `high` -> `xhigh` itself, so values pass straight through) and streams
SSE untouched. With it running, prompt tokens track the setting: `low` 43, `medium` 17, `high` 55.
Setup in [Section 8 of the README](../README.md#section-8-connecting-claude-code).

**`API Usage Billing` in the banner is wrong too - nothing is billed.** That label is Claude Code's
auth-mode readout, triggered by `ANTHROPIC_AUTH_TOKEN` being set; it does not know the endpoint is
local. Worse, cost figures are *fabricated*: sessions against `qwen3.8-27b` record a `costUSD`
computed from a fallback rate with `hasUnknownModelCost: true` (one measured session billed
$0.0102 for 743 in / 260 out tokens that cost nothing). Ignore `/cost`, the status-line cost and
any cost telemetry when pointed at this server.

**Practical setup:** server default `medium` covers Claude Code, Qwen Code and OpenCode; drop
to `low` per-agent in OpenCode for tool-heavy loops; use curl with `xhigh` for one genuinely
hard question.
