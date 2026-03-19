# OpenRecall Tasks

## Current Task: Dependency and Embedding Strategy

- [ ] Replace the mandatory `sentence-transformers` dependency path with an
  optional stack.
- [ ] Document and test fallback behavior so OpenRecall capture/search remains
  functional when `sentence-transformers` is not installed.
- [ ] Add explicit migration guidance for environments that currently rely on
  existing semantic similarity quality but need lighter dependencies.
