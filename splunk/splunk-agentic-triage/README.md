# Agentic Security Triage: Splunk ES + MCP + n8n

Automated security triage that connects Splunk Enterprise Security to an n8n AI Agent via the Splunk MCP Server add-on. When a notable or risk event fires in Splunk ES, the agent gathers context using MCP tools and produces a structured triage report — all on-premises with a local LLM.

## Architecture

```
Splunk ES (lab: brute-force VPN correlation rule)
  └─ notable / risk event fires
       └─ alert action: HTTP POST → http://localhost:5678/webhook/splunk
            └─ Normalize Payload
                 └─ AI Agent (OpenAI-compat → host Mac Ollama / qwen2.5:7b)
                      ├─ Splunk MCP Client (httpStreamable + Bearer)
                      │    → splunk_run_query, splunk_get_metadata, …
                      └─ Debug Agent Output
                           └─ Diagnose MCP & Parse Report
                                └─ MCP OK? → Triage Report | Diagnostic Response
```

| Component | Role | Runs via |
|---|---|---|
| Splunk Enterprise Security | Generates notable/risk events | External |
| Splunk MCP Server | Exposes Splunk as MCP tools (streamable HTTP) | Splunk add-on |
| n8n | Webhook + AI Agent + debug/diagnose path | Docker |
| Ollama (`qwen2.5:7b`) | Local LLM inference (**required** model) | **Native on Mac** (not Docker) |

## Prerequisites

1. **Docker Desktop** (or Docker Engine + Compose) for the n8n container.
2. **Ollama for macOS** installed natively ([download](https://ollama.com/download) or `brew install ollama`).
3. **Splunk MCP Server app** on your Splunk ES instance ([Splunkbase](https://splunkbase.splunk.com/)). Endpoint is typically `https://<host>:8089/services/mcp`.
4. **Encrypted MCP Bearer token** from the **Splunk MCP Server app** (not a regular Settings → Tokens API token). See [Connecting to the MCP Server](https://help.splunk.com/en/splunk-cloud-platform/mcp-server-for-splunk-platform/1.2/connecting-to-the-mcp-server-and-settings).

## Setup

### 1. Install and run Ollama on the Mac

```bash
# App: https://ollama.com/download   — or —
brew install ollama
brew services start ollama   # or: open -a Ollama

curl -s http://127.0.0.1:11434/api/tags
ollama pull qwen2.5:7b       # required model
```

**`qwen2.5:7b` is required** for reliable tool calling with this workflow. Do not use `llama3` / `llama3.2` here — they often print tool intents as text instead of structured function calls.

If you previously ran Ollama in Compose, remove that container so it does not bind port 11434:

```bash
docker rm -f ollama 2>/dev/null || true
```

### 2. Configure environment

```bash
cp .env.example .env
```

```env
SPLUNK_MCP_ENDPOINT=https://host.docker.internal:8089/services/mcp
SPLUNK_TOKEN=your_encrypted_mcp_token
# n8n (Docker) → Ollama on the Mac host
OLLAMA_BASE_URL=http://host.docker.internal:11434
```

### 3. Start n8n

```bash
docker compose up -d
```

Only **n8n** is in Compose. `n8n/entrypoint.sh` on each start:

1. Renders `splunk-triage.json` from the template (`SPLUNK_MCP_ENDPOINT`)
2. Imports credentials (**Ollama Local**, **Ollama OpenAI Compat** at `…/v1`, **Splunk MCP Bearer Auth**)
3. **Duplicate handling** — imports the workflow only if `splunk-triage` / `Splunk Agentic Triage` is missing (avoids duplicate copies and preserves executions)
4. **Auto-activation** — seeds `workflow_history` if needed, sets `active=1`, ensures `POST /webhook/splunk` is registered
5. Waits until `/webhook/*` routes are mounted (avoids HTML `Cannot POST /webhook/splunk`)

No manual import/toggle is required on first boot.

`NODE_TLS_REJECT_UNAUTHORIZED=0` is set so the MCP Client can use Splunk’s self-signed cert (**lab only**). Prefer `NODE_EXTRA_CA_CERTS` or a trusted CA in production.

### 4. Confirm the webhook

```text
http://localhost:5678/webhook/splunk
```

Workflow path includes **Debug Agent Output** (logs `intermediateSteps`) and **Diagnose MCP & Parse Report** (sets `diagnostics` / `route_ok`). Inspect logs with:

```bash
docker logs n8n 2>&1 | grep -A 50 'AI Agent DEBUG'
```

### 5. Splunk ES alert / lab rule

Wire notables to `http://<n8n-host>:5678/webhook/splunk` (POST). Lab app `splunk/agentic_triage_test/` synthesizes a **VPN brute-force / password-spray** notable (`makeresults`, no indexed data) with fields such as `failed_attempts`, `dest_user`, `attack_type`, and timestamps. See that app’s README.

## MCP connection

| Setting | Value |
|---|---|
| Endpoint | `SPLUNK_MCP_ENDPOINT` |
| Transport | `httpStreamable` |
| Auth | Bearer (encrypted MCP token) |

Primary tools: `splunk_run_query`, `splunk_get_metadata`, plus other `splunk_*` tools — see `specs/mcp-tools.md`.

## Testing

```bash
curl -X POST http://localhost:5678/webhook/splunk \
  -H "Content-Type: application/json" \
  -d '{
    "event_id": "bf-manual-001",
    "severity": "high",
    "src_ip": "203.0.113.87",
    "dest_user": "jsmith",
    "failed_attempts": 12,
    "attack_type": "password_spray_brute_force"
  }'
```

## Project structure

```
├── docker-compose.yml         # n8n only
├── .env.example
├── README.md
├── n8n/
│   ├── entrypoint.sh          # credentials, duplicate-safe import, auto-activate
│   └── workflows/
│       ├── splunk-triage.json.template
│       └── splunk-triage.json
├── splunk/agentic_triage_test/  # brute-force VPN lab correlation rule
└── specs/
```

## Notes

- Ollama is **native on Mac**; n8n uses `http://host.docker.internal:11434` (+ `/v1` OpenAI-compat)
- **Required model:** `qwen2.5:7b` (workflow timeout 600s; `maxTokens` 1024)
- Debug/diagnose nodes surface MCP tool usage in the webhook JSON and container logs
- Entrypoint: duplicate-safe import + auto-activation; existing workflows/executions are kept on recreate
