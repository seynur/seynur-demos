# Test Correlation Rule: Agentic Triage Notable

Lab Splunk ES correlation search that synthesizes a **VPN brute-force / password-spray** notable every 5 minutes. It uses `| makeresults` only — no indexed sample data — and emits rich fields so the n8n agent can write a meaningful triage report.

## App layout

```
splunk/agentic_triage_test/
├── local/
│   ├── app.conf
│   └── savedsearches.conf
└── metadata/
    └── local.meta
```

## What it does

| Item | Value |
|---|---|
| Search name | `Test - Agentic Triage Notable` |
| Schedule | Every 5 minutes |
| Scenario | 12 synthetic failed VPN logins → 1 aggregated notable |
| MITRE | T1110.001 (Password Guessing) |
| Notable severity | high |
| `src_ip` | `203.0.113.87` (attacker) |
| `dest_ip` / `dest_host` | `10.0.1.25` / `corp-vpn-gw01.internal.example` |
| `dest_user` | `jsmith` |
| `failed_attempts` | `12` |
| `attack_type` | `password_spray_brute_force` |

Notable / webhook payload includes context fields such as `failed_attempts`, `first_seen`, `last_seen`, `attack_window`, `failure_reasons`, `mitre_technique`, `ioc_summary`, and `recommended_actions`, plus the n8n contract fields:

```json
{
  "event_id": "...",
  "severity": "high",
  "src_ip": "203.0.113.87",
  "dest_user": "jsmith",
  "failed_attempts": 12,
  "attack_type": "password_spray_brute_force"
}
```

## Install

1. Copy the app onto the ES search head:

   ```bash
   cp -R splunk/agentic_triage_test $SPLUNK_HOME/etc/apps/
   ```

2. Fix ownership if needed (Linux):

   ```bash
   chown -R splunk:splunk $SPLUNK_HOME/etc/apps/agentic_triage_test
   ```

3. Restart Splunk or reload search:

   ```bash
   $SPLUNK_HOME/bin/splunk restart
   # or:
   $SPLUNK_HOME/bin/splunk reload search
   ```

4. In Splunk ES go to **Content → Content Management**, find **Test - Agentic Triage Notable**, and confirm it is **Enabled**.

## Wire the alert to n8n

The webhook action is already defined in `local/savedsearches.conf`:

```conf
action.webhook = 1
action.webhook.param.url = http://localhost:5678/webhook/splunk
```

Splunk’s built-in webhook POSTs the first result row under a `result` object (plus `sid`, `app`, etc.). The search already outputs `event_id`, `severity`/`urgency`, and `src_ip`, so those fields are included automatically. The n8n workflow’s **Normalize Payload** node unwraps that envelope.

This URL assumes Splunk runs on the host (not in Docker) and n8n is published on port 5678. If n8n is elsewhere, edit the URL in `savedsearches.conf` (or in the search’s alert actions UI) and reload Splunk.

## Verify

1. Wait for the next 5-minute cron, or run the search manually once from **Search**.
2. In ES **Incident Review**, confirm a notable titled like `Brute Force Detected: jsmith from 203.0.113.87`.
3. Confirm n8n received the webhook (Executions view) and the agent started triage with the brute-force context fields.

## Manual one-shot test (no wait)

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

## Disable / remove

- Disable the search in Content Management, or set `disabled = true` in `local/savedsearches.conf` and reload.
- To remove the app: delete `$SPLUNK_HOME/etc/apps/agentic_triage_test` and restart Splunk.

## Notes

- `| makeresults` is intentional for lab use — it does not depend on indexed data.
- Do not enable this search in production; it will create a notable every 5 minutes.
