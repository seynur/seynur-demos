#!/bin/sh
# Prepare credentials + workflow endpoint from .env, then start n8n.
set -e

OLLAMA_BASE_URL="${OLLAMA_BASE_URL:-http://host.docker.internal:11434}"
# OpenAI-compat base is always OLLAMA_BASE_URL + /v1 (host Mac Ollama by default).
OLLAMA_OPENAI_BASE="${OLLAMA_BASE_URL%/}/v1"
WORKFLOW_DIR="/home/node/.n8n/workflows"
TEMPLATE_FILE="${WORKFLOW_DIR}/splunk-triage.json.template"
WORKFLOW_FILE="${WORKFLOW_DIR}/splunk-triage.json"
WORKFLOW_NAME="Splunk Agentic Triage"
WORKFLOW_ID="splunk-triage"
WEBHOOK_PATH="splunk"
WEBHOOK_NODE="Webhook"
DB_FILE="/home/node/.n8n/database.sqlite"
SQLITE_MODULE="/usr/local/lib/node_modules/n8n/node_modules/sqlite3"
CREDS_FILE="/tmp/splunk-triage-credentials.json"

if [ -z "$SPLUNK_MCP_ENDPOINT" ]; then
  echo "WARNING: SPLUNK_MCP_ENDPOINT is not set."
fi

if [ -z "$SPLUNK_TOKEN" ]; then
  echo "WARNING: SPLUNK_TOKEN is not set (use an encrypted MCP token from the Splunk MCP Server app)."
fi

# Always generate the importable workflow from the template so the Endpoint
# field is a concrete URL (not an undefined $env expression).
if [ -f "$TEMPLATE_FILE" ]; then
  if [ -n "$SPLUNK_MCP_ENDPOINT" ]; then
    # Use # delimiters — URLs contain /, so never use / as the sed separator
    ESCAPED_ENDPOINT=$(printf '%s' "$SPLUNK_MCP_ENDPOINT" | sed 's#[\&]#\\&#g')
    sed "s#__SPLUNK_MCP_ENDPOINT__#${ESCAPED_ENDPOINT}#g" "$TEMPLATE_FILE" > "$WORKFLOW_FILE"
    echo "Generated workflow with MCP endpoint: $SPLUNK_MCP_ENDPOINT"
  else
    cp "$TEMPLATE_FILE" "$WORKFLOW_FILE"
    echo "WARNING: Generated workflow still contains __SPLUNK_MCP_ENDPOINT__ placeholder."
  fi
fi

# Decrypted credential payload — n8n encrypts on import with the instance key
cat > "$CREDS_FILE" <<EOF
[
  {
    "id": "ollama-local",
    "name": "Ollama Local",
    "type": "ollamaApi",
    "data": {
      "baseUrl": "${OLLAMA_BASE_URL}",
      "apiKey": ""
    }
  },
  {
    "id": "ollama-openai",
    "name": "Ollama OpenAI Compat",
    "type": "openAiApi",
    "data": {
      "apiKey": "ollama",
      "url": "${OLLAMA_OPENAI_BASE}"
    }
  },
  {
    "id": "splunk-mcp-bearer",
    "name": "Splunk MCP Bearer Auth",
    "type": "httpBearerAuth",
    "data": {
      "token": "${SPLUNK_TOKEN}"
    }
  }
]
EOF

echo "Importing Ollama credentials (native=${OLLAMA_BASE_URL}, openai=${OLLAMA_OPENAI_BASE}) and Splunk MCP Bearer Auth..."
n8n import:credentials --input="$CREDS_FILE" 2>/dev/null || \
  echo "Credential import skipped or partially applied (credentials may already exist)."

rm -f "$CREDS_FILE"

# Preserve n8n data across restarts: only import the workflow when it is
# missing. Never delete workflows, webhooks, or executions on boot.
workflow_exists() {
  if [ ! -f "$DB_FILE" ]; then
    return 1
  fi
  WORKFLOW_NAME="$WORKFLOW_NAME" WORKFLOW_ID="$WORKFLOW_ID" \
    DB_FILE="$DB_FILE" SQLITE_MODULE="$SQLITE_MODULE" node <<'NODE'
const sqlite3 = require(process.env.SQLITE_MODULE);
const db = new sqlite3.Database(process.env.DB_FILE, sqlite3.OPEN_READONLY);
db.get(
  'SELECT id FROM workflow_entity WHERE id = ? OR name = ? LIMIT 1',
  [process.env.WORKFLOW_ID, process.env.WORKFLOW_NAME],
  (err, row) => {
    db.close();
    if (err) {
      console.error('Workflow existence check failed:', err.message);
      process.exit(2);
    }
    process.exit(row ? 0 : 1);
  },
);
NODE
}

# import:workflow leaves workflows inactive and does not create a
# workflow_history row, so update:workflow --active=true fails with a FK
# error. Seed history, mark active, and upsert the webhook path so
# LiveWebhooks can resolve it as soon as Express mounts /webhook/* (n8n
# listens and logs "ready" BEFORE routes and Start Active Workflows run).
ensure_workflow_active() {
  if [ ! -f "$DB_FILE" ]; then
    echo "WARNING: Database not found; cannot activate workflow."
    return 1
  fi
  echo "Ensuring workflow ${WORKFLOW_ID} is active with webhook /${WEBHOOK_PATH}..."
  WORKFLOW_NAME="$WORKFLOW_NAME" WORKFLOW_ID="$WORKFLOW_ID" \
    WEBHOOK_PATH="$WEBHOOK_PATH" WEBHOOK_NODE="$WEBHOOK_NODE" \
    DB_FILE="$DB_FILE" SQLITE_MODULE="$SQLITE_MODULE" node <<'NODE'
const sqlite3 = require(process.env.SQLITE_MODULE);
const db = new sqlite3.Database(process.env.DB_FILE);
const workflowId = process.env.WORKFLOW_ID;
const workflowName = process.env.WORKFLOW_NAME;
const webhookPath = process.env.WEBHOOK_PATH;
const webhookNode = process.env.WEBHOOK_NODE;

db.serialize(() => {
  db.run('PRAGMA foreign_keys = ON');
  db.get(
    'SELECT id, versionId, name, nodes, connections, description, active FROM workflow_entity WHERE id = ? OR name = ? LIMIT 1',
    [workflowId, workflowName],
    (err, wf) => {
      if (err) {
        console.error('Workflow activation failed:', err.message);
        process.exit(1);
      }
      if (!wf) {
        console.error(`Workflow activation failed: workflow "${workflowId}" not found.`);
        process.exit(1);
      }

      db.run(
        `INSERT INTO workflow_history
          (versionId, workflowId, authors, nodes, connections, name, autosaved, description)
         VALUES (?, ?, ?, ?, ?, ?, ?, ?)
         ON CONFLICT(versionId) DO NOTHING`,
        [
          wf.versionId,
          wf.id,
          'entrypoint',
          wf.nodes,
          wf.connections,
          wf.name,
          0,
          wf.description || null,
        ],
        (historyErr) => {
          if (historyErr) {
            console.error('Workflow activation failed:', historyErr.message);
            process.exit(1);
          }
          db.run(
            'UPDATE workflow_entity SET active = 1, activeVersionId = COALESCE(activeVersionId, ?) WHERE id = ?',
            [wf.versionId, wf.id],
            (activateErr) => {
              if (activateErr) {
                console.error('Workflow activation failed:', activateErr.message);
                process.exit(1);
              }
              db.run(
                `INSERT INTO webhook_entity
                  (workflowId, webhookPath, method, node, webhookId, pathLength)
                 VALUES (?, ?, 'POST', ?, NULL, NULL)
                 ON CONFLICT(webhookPath, method) DO UPDATE SET
                   workflowId = excluded.workflowId,
                   node = excluded.node`,
                [wf.id, webhookPath, webhookNode],
                function (hookErr) {
                  if (hookErr) {
                    console.error('Webhook registration failed:', hookErr.message);
                    process.exit(1);
                  }
                  console.log(
                    `Workflow ${wf.id} active (version ${wf.versionId}) with POST /webhook/${webhookPath}.`,
                  );
                  db.close();
                },
              );
            },
          );
        },
      );
    },
  );
});
NODE
}

if [ -f "$WORKFLOW_FILE" ]; then
  if workflow_exists; then
    echo "Workflow \"${WORKFLOW_NAME}\" already exists — skipping import (preserving executions and data)."
    ensure_workflow_active || true
  else
    echo "Workflow not found — importing from ${WORKFLOW_FILE}..."
    n8n import:workflow --input="$WORKFLOW_FILE" || \
      echo "WARNING: Workflow import failed."
    ensure_workflow_active || true
  fi
else
  echo "WARNING: Workflow file not found at ${WORKFLOW_FILE}; skipping import."
fi

# n8n calls listen() and logs "ready" before mounting /webhook/* routes.
# Start in the background, wait until the production webhook is reachable,
# then keep n8n as PID 1 via wait.
if [ "$1" = "start" ]; then
  n8n start &
  N8N_PID=$!

  echo "Waiting for n8n to mount /webhook/* routes..."
  WEBHOOK_PATH="$WEBHOOK_PATH" DB_FILE="$DB_FILE" SQLITE_MODULE="$SQLITE_MODULE" node <<'NODE'
const sqlite3 = require(process.env.SQLITE_MODULE);
const webhookPath = process.env.WEBHOOK_PATH;
const probeUrl = 'http://127.0.0.1:5678/webhook/__entrypoint_boot_check__';
const maxAttempts = 60;

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function webhookRowExists() {
  return new Promise((resolve, reject) => {
    const db = new sqlite3.Database(process.env.DB_FILE);
    db.get(
      `SELECT 1 AS ok FROM webhook_entity WHERE webhookPath = ? AND method = 'POST'`,
      [webhookPath],
      (err, row) => {
        db.close();
        if (err) reject(err);
        else resolve(Boolean(row));
      },
    );
  });
}

(async () => {
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      const registered = await webhookRowExists();
      const res = await fetch(probeUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: '{}',
        signal: AbortSignal.timeout(2000),
      });
      const text = await res.text();
      // Before start() mounts routes, Express returns HTML "Cannot POST".
      // After routes are mounted, unknown paths return JSON "not registered".
      const routesMounted =
        text.includes('not registered') && !text.includes('Cannot POST');
      if (registered && routesMounted) {
        console.log(`Webhook routes ready after ${attempt} attempt(s).`);
        process.exit(0);
      }
    } catch {
      // Connection refused / timeout while n8n is still booting.
    }
    await sleep(500);
  }
  console.error('WARNING: Timed out waiting for webhook routes; continuing anyway.');
  process.exit(0);
})();
NODE

  wait "$N8N_PID"
else
  exec n8n "$@"
fi
