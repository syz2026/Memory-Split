# Current-source fixtures

`tests/test_current_sources.py` builds tiny Wikidata tarballs and pinned local
Git repositories from deterministic in-memory ARC-style tasks. The generated
fixtures exercise the same archive and Git staging paths as production while
remaining fully offline and small enough to inspect in a temporary directory.
