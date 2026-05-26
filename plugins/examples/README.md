# Facet Plugin Examples

Runnable plugin reference implementations. Copy a file into the project's
top-level `plugins/` directory (drop the `.example` suffix) and enable
plugins in `scoring_config.json`:

```json
"plugins": {
  "enabled": true,
  "webhooks": [],
  "actions": {}
}
```

## Available examples

| File | Trigger | What it does |
|------|---------|--------------|
| `slack_webhook.py.example` | `on_high_score` | Sends a formatted Slack message via the official Slack webhook URL when a photo scores above your configured threshold. |
| `copy_to_folder.py.example` | `on_high_score` | Copies every photo above the threshold into a destination folder, preserving the directory structure. Handy for building a "best of" folder for backup or print. |
| `score_publisher.py.example` | `on_score_complete` | Writes each finished score as a JSON line to a rolling log file. Use with `tail -F` for a live feed, or feed into ELK/Loki for aggregation. |

## Webhook payload shape

Outgoing HTTP webhooks (configured under `plugins.webhooks` in
`scoring_config.json`) POST a JSON body of the form:

```json
{
  "event": "on_high_score",
  "data": {
    "path": "/photos/2026/01/IMG_1234.cr3",
    "aggregate": 9.4,
    "aesthetic": 8.9,
    "comp_score": 9.1,
    "category": "portrait",
    "tags": ["portrait", "golden_hour"]
  }
}
```

`event` is one of:

* `on_score_complete` — every photo, once scoring finishes
* `on_new_photo` — first time a photo is added to the DB
* `on_burst_detected` — when a burst group is identified
* `on_high_score` — only when `aggregate >= min_score` (default 8.0,
  configurable per webhook)

A formal JSON Schema for the payload lives in
`plugins/examples/webhook_payload.schema.json` if you want to validate
incoming requests on the receiving side.

## Security notes

* Webhook URLs are validated against a private-network deny-list at
  registration *and* delivery time. Loopback, RFC1918, link-local
  (including AWS metadata 169.254.169.254), and unsupported schemes
  (ftp, file, gopher) are rejected with a logged error.

* The HTTP request goes to the resolved IP, but the `Host` header
  carries the original hostname — DNS rebinding cannot redirect a
  validated URL to a private address mid-flight.

* Delivery is best-effort: a 5xx response, timeout, or transport
  error is logged but does not retry. If you need at-least-once
  semantics, build a queue on the receiving side and ack via your
  own retry policy.

* Outbound timeout is 10 seconds. Plan your receiver accordingly.
