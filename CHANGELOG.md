# Changelog

## Unreleased

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
