# OpenRecall Tasks

## Current Task: Dependency and Embedding Strategy

- [ ] Replace the mandatory `sentence-transformers` dependency path with an
  optional stack.
- [ ] Document and test fallback behavior so OpenRecall capture/search remains
  functional when `sentence-transformers` is not installed.
- [ ] Add explicit migration guidance for environments that currently rely on
  existing semantic similarity quality but need lighter dependencies.
- [x] Add a producer-owned normalized capture sidecar helper so screenshot
  captures can participate in the suite-level `itir.normalized.artifact.v1`
  contract without replacing canonical storage.
- [x] Add a bounded non-authoritative search-follow artifact plus normalized
  derived wrapper so OpenRecall retrieval stays advisory and reusable across
  the suite.
