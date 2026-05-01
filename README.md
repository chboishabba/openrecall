```
   ____                   ____                  ____   
  / __ \____  ___  ____  / __ \___  _________ _/ / /   
 / / / / __ \/ _ \/ __ \/ /_/ / _ \/ ___/ __ `/ / /    
/ /_/ / /_/ /  __/ / / / _, _/  __/ /__/ /_/ / / /     
\____/ .___/\___/_/ /_/_/ |_|\___/\___/\__,_/_/_/      
    /_/                                                                                                                         
```
**Enjoy this project?** Show your support by starring it! ⭐️ Thank you!

Join our [Discord](https://discord.gg/RzvCYRgUkx) and/or [Telegram](https://t.me/+5DULWTesqUYwYjY0) community to stay informed of updates!

# Take Control of Your Digital Memory

OpenRecall is a fully open-source, privacy-first alternative to proprietary solutions like Microsoft's Windows Recall or Limitless' Rewind.ai. With OpenRecall, you can easily access your digital history, enhancing your memory and productivity without compromising your privacy.

## What does it do?

OpenRecall captures your digital history through regularly taken snapshots, which are essentially screenshots. The text and images within these screenshots are analyzed and made searchable, allowing you to quickly find specific information by typing relevant keywords into OpenRecall. You can also manually scroll back through your history to revisit past activities.

https://github.com/openrecall/openrecall/assets/16676419/cfc579cb-165b-43e4-9325-9160da6487d2

## Why Choose OpenRecall?

OpenRecall offers several key advantages over closed-source alternatives:

- **Transparency**: OpenRecall is 100% open-source, allowing you to audit the source code for potential backdoors or privacy-invading features.
- **Cross-platform Support**: OpenRecall works on Windows, macOS, and Linux, giving you the freedom to use it on your preferred operating system.
- **Privacy-focused**: Your data is stored locally on your device, no internet connection or cloud is required. In addition, you have the option to encrypt the data on a removable disk for added security, read how in our [guide](docs/encryption.md) here. 
- **Hardware Compatibility**: OpenRecall is designed to work with a [wide range of hardware](docs/hardware.md), unlike proprietary solutions that may require specific certified devices.

<p align="center">
  <a href="https://twitter.com/elonmusk/status/1792690964672450971" target="_blank">
    <img src="images/black_mirror.png" alt="Elon Musk Tweet" width="400">
  </a>
</p>

## Features

- **Time Travel**: Revisit and explore your past digital activities seamlessly across Windows, macOS, or Linux.
- **Local-First AI**: OpenRecall harnesses the power of local AI processing to keep your data private and secure.
- **Semantic Search**: Advanced local OCR interprets your history, providing robust semantic search capabilities.
- **Full Control Over Storage**: Your data is stored locally, giving you complete control over its management and security.

## Embedding Backend

- OpenRecall can generate text embeddings for search using an optional
  `sentence-transformers` backend.
- By default, the app attempts a local `sentence-transformers` model and falls
  back to a deterministic local vectorizer if that dependency is unavailable.
- To force the local fallback path for lighter/isolated installs, set:

```bash
export OPENRECALL_EMBEDDING_BACKEND=local
```

- To avoid optional model downloads entirely, install OpenRecall without the
  embedding extras (recommended for CPU-only or constrained environments).
  If you want higher-quality semantic recall, install the optional
  dependency set.

```bash
pip install .
pip install .[embeddings]
```

## Optional ITIR MCP bridge

Within `ITIR-suite`, OpenRecall can also hand bounded capture observations to
the suite-level `itir-mcp` contract for deterministic comparison/coherence
analysis.

- local OCR, screenshot capture, and image similarity remain OpenRecall-owned
- comparison/coherence semantics can be delegated to `itir.compare_observations`
  and `itir.score_coherence`
- MCP is the canonical tool contract; the current local Python client uses the
  `itir-mcp --bridge` JSON transport as its concrete invocation path
- the public comparison helpers consume the guarded `safe_call` path and return a
  bounded policy hint (`status_explanation` + `policy_hint`) with any
  verification outcome
- verified outcomes keep the previous `itir.compare_observations` behavior
  (result payload present and used as before)
- rejected/abstained outcomes now return an explicit non-judgmental signal instead
  of raising through OpenRecall APIs
- this keeps one canonical comparison contract across OpenRecall and
  WorldMonitor instead of duplicating comparison logic in each producer

## OCR backend and offline processing

OpenRecall's default path still runs OCR inline with live screenshot capture,
but the OCR surface is now configurable so heavier ML can move out of the hot
path when needed.

- `OPENRECALL_OCR_BACKEND=local` keeps the existing local Doctr OCR path.
- `OPENRECALL_OCR_BACKEND=http` sends the image to an external OCR API instead.
- `OPENRECALL_OCR_API_URL` points at that OCR service.
- `OPENRECALL_OCR_API_TOKEN` optionally adds `Authorization: Bearer ...`.
- `OPENRECALL_OCR_API_HEADERS` optionally adds extra JSON-encoded headers.
- `--capture-only` records screenshots without running OCR or embeddings in the
  live loop, which is useful when you want to defer OCR, segmentation, or other
  heavy ML until a later offline batch.

The HTTP OCR seam is intentionally simple: POST a JSON body with
`image_base64` and `mime_type`, and return a JSON object containing at least
`text`, plus optional `segments` and `metadata`. That lets an external worker,
GPU box, or hosted API own OCR/segmentation complexity while OpenRecall keeps
capture/storage local.

### Offline OCR processing

To process saved screenshots after capture:

```bash
python -m openrecall.ocr_jobs process ~/.local/share/openrecall/screenshots \
  --backend http \
  --output openrecall_ocr_results.json
```

That emits one JSON record per image with text, segment count, backend, and
elapsed seconds.

### OCR benchmarking

To measure OCR throughput and basic text accuracy, create a manifest JSON file:

```json
[
  {
    "image_path": "fixtures/receipt.png",
    "expected_text": "TOTAL 12.50"
  }
]
```

Then run:

```bash
python -m openrecall.ocr_jobs benchmark \
  --manifest benchmark_manifest.json \
  --backend local
```

The benchmark output reports:

- `chars_per_second`
- `exact_match_rate`
- `char_error_rate`
- per-image elapsed time and normalized similarity

This is a lightweight runtime/quality gate rather than a formal OCR research
benchmark, but it is enough to compare local vs external OCR paths and spot
heavy-model regressions.

## Session bundles for secure central processing

If you want endpoints to capture locally but do OCR, segmentation, or heavier
ML centrally, OpenRecall now supports a compressed day/session bundle format.

The local machine can package one day of captures plus metadata:

```bash
python -m openrecall.bundle_jobs export-day \
  --date 2026-05-01 \
  --session-id alice-home-2026-05-01 \
  --org-id personal \
  --user-id alice \
  --device-id workstation-a \
  --output alice-home-2026-05-01.orbundle.zip \
  --retention-days 30 \
  --signing-secret "$OPENRECALL_BUNDLE_SECRET" \
  --compression lzma
```

That bundle contains:

- `manifest.json` with org, user, device, session, host, date, and capture metadata
- the captured screenshots for that window under `captures/`
- optional retention metadata (`retention_days`, `delete_after`)
- optional HMAC signature when `--signing-secret` is provided

On the secure processing server, ingest the bundle into a central registry:

```bash
python -m openrecall.bundle_jobs ingest-bundle \
  --bundle alice-home-2026-05-01.orbundle.zip \
  --signing-secret "$OPENRECALL_BUNDLE_SECRET" \
  --registry-root /srv/openrecall-processing
```

The registry keeps track of multiple bundles across:

- organization
- user
- device
- session
- captured date

The central registry and timeline live in one shared SQLite store, but all
history queries are explicitly scoped by `org_id` and `user_id`. That means the
storage can be shared without forcing one global default history.

### Multi-user default model

This bundle/registry shape is multi-user, but it does not force users into one
shared history by default.

The primary story is one person federating memory across their own machines, or
across a small self-managed household/homelab. Corporate or institutional use
is possible, but it is not the normative framing for the feature and it does
not relax the suite's anti-panopticon boundary.

What it supports now:

- each endpoint/user can keep a separate local `recall.db`
- each exported bundle carries:
  - `org_id`
  - `user_id`
  - `device_id`
  - `session_id`
  - `host_id`
  - `captured_date`
- the central ingest registry stores bundles and timeline entries keyed by
  those identifiers in one shared DB

So the default operating model is:

- local capture/search history remains per-user
- central processing can handle many users
- the central registry tracks bundles separately by org, user, device, and session
- OCR write-back goes back into a chosen local `recall.db`, not one global
  merged history

For easier merge and split semantics, both the local `entries` table and the
central timeline now keep:

- `captured_date` for day-bucket grouping, export windows, retention, and
  coarse merge/sync boundaries
- `timestamp` / `captured_at` for precise ordering and replay inside that day

### Unified timeline query

To inspect the shared central timeline for one user across multiple devices:

```bash
python -m openrecall.bundle_jobs query-timeline \
  --registry-root /srv/openrecall-processing \
  --org-id personal \
  --user-id alice
```

To focus on one machine only:

```bash
python -m openrecall.bundle_jobs query-timeline \
  --registry-root /srv/openrecall-processing \
  --org-id personal \
  --user-id alice \
  --device-id workstation-a
```

To query one day bucket directly:

```bash
python -m openrecall.bundle_jobs query-timeline \
  --registry-root /srv/openrecall-processing \
  --org-id personal \
  --user-id alice \
  --captured-date 2026-05-01
```

The default timeline semantics are:

- keep each device event separate
- preserve device provenance
- order chronologically
- allow later replay to focus one device or view several devices side by side
- never auto-merge overlapping captures from different devices in v1

That means the current flow supports one user's cross-device replay and, where
explicitly enabled, larger shared processing deployments without immediately
collapsing everyone into a single recall stream.

What it could also support later, if you explicitly want it:

- an org-wide aggregate processing/reporting layer on top of the central
  registry
- a cross-user search/index layer

Those would be extra policy/data-product layers, not the default behavior of
the current bundle plus write-back flow.

OpenRecall does not treat a shared central registry as permission to create an
authoritative, merged institutional memory. Default queries are scoped by
`org_id` and `user_id`, per-device provenance stays visible, and cross-device
views remain chronological observer surfaces rather than hidden interpretive
upgrades.

This gives you a simple capture-now / process-centrally-later workflow without
requiring every endpoint to host the heavy OCR/ML stack locally.

### Writing processed OCR back into `recall.db`

After remote OCR completes, write the results back into the local OpenRecall DB
using the original session manifest plus the OCR results JSON:

```bash
python -m openrecall.bundle_jobs writeback-ocr \
  --manifest manifest.json \
  --ocr-results openrecall_ocr_results.json \
  --db-path ~/.local/share/openrecall/recall.db
```

This groups OCR results back onto their original capture entries by screenshot
filename and upserts the final search entries into `recall.db`.

If the remote processor returns extra payloads, the write-back path now also
supports:

- remote embeddings via `embedding`
- richer vision payloads via `segments` and `metadata`

When a remote embedding is present and shaped correctly, OpenRecall uses it
instead of rebuilding the embedding locally. Vision payloads are preserved as
JSON sidecars alongside the manifest directory so later segmentation/layout
workflows can reuse them.

For self-managed deployments, the remote vision contract is intentionally broad.
That means a trusted local processor can return richer payloads, but OpenRecall
still treats them as observer-class outputs. They do not become hidden
authority, person scoring, or cross-context identity resolution by default.

That completes the loop:

- capture locally
- process OCR/ML elsewhere
- search locally later

<p align="center">
  <img src="images/lisa_rewind.webp" alt="Lisa Rewind" width="400">
</p>


## Comparison



| Feature          | OpenRecall                    | Windows Recall                                  | Rewind.ai                              |
|------------------|-------------------------------|--------------------------------------------------|----------------------------------------|
| Transparency     | Open-source                   | Closed-source                                    | Closed-source                          |
| Supported Hardware | All                         | Copilot+ certified Windows hardware              | M1/M2 Apple Silicon                    |
| OS Support       | Windows, macOS, Linux         | Windows                                          | macOS                                  |
| Privacy          | On-device, self-hosted        | Microsoft's privacy policy applies               | Connected to ChatGPT                   |
| Cost             | Free                          | Part of Windows 11 (requires specialized hardware) | Monthly subscription                   |

## Quick links
- [Roadmap](https://github.com/orgs/openrecall/projects/2) and you can [vote for your favorite features](https://github.com/openrecall/openrecall/discussions/9#discussion-6775473)
- [FAQ](https://github.com/openrecall/openrecall/wiki/FAQ)

## Get Started

### Prerequisites
- Python 3.11
- MacOSX/Windows/Linux
- Git

To install:
```
python3 -m pip install --upgrade --no-cache-dir git+https://github.com/openrecall/openrecall.git
```

To run:
```
python3 -m openrecall.app
```
Open your browser to:
[http://localhost:8082](http://localhost:8082) to access OpenRecall.

## Arguments
`--storage-path` (default: user data path for your OS): allows you to specify the path where the screenshots and database should be stored. We recommend [creating an encrypted volume](docs/encryption.md) to store your data.

`--primary-monitor-only` (default: False): only record the primary monitor (rather than individual screenshots for other monitors)

`--capture-only` (default: False): save screenshots but skip OCR and embeddings
in the live loop so OCR/segmentation can run later in offline jobs or on a
separate machine.

## Uninstall instructions

To uninstall OpenRecall and remove all stored data:

1. Uninstall the package:
   ```
   python3 -m pip uninstall openrecall
   ```

2. Remove stored data:
   - On Windows:
     ```
     rmdir /s %APPDATA%\openrecall
     ```
   - On macOS:
     ```
     rm -rf ~/Library/Application\ Support/openrecall
     ```
   - On Linux:
     ```
     rm -rf ~/.local/share/openrecall
     ```

Note: If you specified a custom storage path at any time using the `--storage-path` argument, make sure to remove that directory too.

## Contribute

As an open-source project, we welcome contributions from the community. If you'd like to help improve OpenRecall, please submit a pull request or open an issue on our GitHub repository.

## Contact the maintainers
mail@datatalk.be

## License

OpenRecall is released under the [AGPLv3](https://opensource.org/licenses/AGPL-3.0), ensuring that it remains open and accessible to everyone.
