# Changelog

## Unreleased

- Make `sentence-transformers` optional in the vendored OpenRecall capture path.
- Add a deterministic local embedding fallback when `sentence-transformers` is not
  installed or when `OPENRECALL_EMBEDDING_BACKEND=local`.
- Remove `sentence-transformers` from hard installation requirements and move it
  into optional extras.
- Add local-embedding test coverage and update zero-vector cosine-similarity
  behavior docs/tests.
