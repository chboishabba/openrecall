# OpenRecall Tasks

## Current Task: Dependency and Embedding Strategy

- [ ] Replace the mandatory `sentence-transformers` dependency path with an
  optional stack.
- [ ] Document and test fallback behavior so OpenRecall capture/search remains
  functional when `sentence-transformers` is not installed.
- [ ] Add explicit migration guidance for environments that currently rely on
  existing semantic similarity quality but need lighter dependencies.
- [x] Add optional write-back/import tooling so offline OCR batch results can
  repopulate the local search DB without re-running live capture.
- [ ] Extend the external OCR seam into a broader external vision seam for
  segmentation/layout/object-detection payloads where downstream workflows need
  more than plain text.
- [ ] Add a reproducible OCR benchmark fixture set that can compare local vs
  remote backends over the same screenshots and ground truth.
- [x] Add optional rehydration/write-back so centrally processed bundle results
  can flow back into local `recall.db` instances when the deployment wants
  local search after cloud processing.
- [x] Add signed bundle manifests and retention/expiry policy controls for
  multi-user secure-cloud processing deployments.
- [x] Accept remote embeddings during OCR write-back instead of always
  rebuilding embeddings locally.
- [x] Preserve richer remote vision payloads (`segments`, `metadata`) during
  write-back so later segmentation/layout workflows can reuse them.
- [ ] Add explicit server-side tenant/RBAC policy for non-self-managed
  deployments so shared central stores do not become a default cross-user
  visibility surface.
- [x] Add a producer-owned normalized capture sidecar helper so screenshot
  captures can participate in the suite-level `itir.normalized.artifact.v1`
  contract without replacing canonical storage.
- [x] Add a bounded non-authoritative search-follow artifact plus normalized
  derived wrapper so OpenRecall retrieval stays advisory and reusable across
  the suite.
