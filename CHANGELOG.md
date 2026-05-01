# Changelog

## Unreleased

- Add a configurable OCR backend seam with local Doctr and external HTTP modes.
- Add `--capture-only` so live screenshot capture can defer OCR/embedding work
  for later offline processing.
- Add `python -m openrecall.ocr_jobs process` and `benchmark` commands for
  offline OCR runs, chars-per-second measurement, and basic accuracy reporting.
- Add focused OCR backend and benchmark tests.
- Add `python -m openrecall.bundle_jobs export-day` and `ingest-bundle` so
  one day/session of captures can be packed into a high-compression bundle and
  tracked centrally by org/user/device/session on a secure processing server.
- Add `python -m openrecall.bundle_jobs query-timeline` so the shared central
  store can render one user's cross-device history without auto-merging device
  provenance.
- Add `python -m openrecall.bundle_jobs writeback-ocr` so offline or remote OCR
  results can be written back into `recall.db` for later local search.
- Add optional HMAC-signed bundle manifests plus retention/expiry metadata for
  multi-user secure-cloud processing flows.
- Add support for richer remote vision payloads (`segments`, `metadata`) and
  optional remote embeddings during OCR write-back.
- Add `openrecall/normalized_artifact.py` and focused tests so captured
  screenshots can emit a producer-owned `itir.normalized.artifact.v1`
  `source_artifact` sidecar without replacing canonical storage.
- Add a bounded non-authoritative OpenRecall search-follow artifact and
  normalized derived wrapper so local search results can participate in the
  suite retrieval/follow posture without becoming canonical truth.
- Make `sentence-transformers` optional in the vendored OpenRecall capture path.
- Add a deterministic local embedding fallback when `sentence-transformers` is not
  installed or when `OPENRECALL_EMBEDDING_BACKEND=local`.
- Remove `sentence-transformers` from hard installation requirements and move it
  into optional extras.
- Add local-embedding test coverage and update zero-vector cosine-similarity
  behavior docs/tests.
