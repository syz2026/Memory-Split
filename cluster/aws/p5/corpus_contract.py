"""Compatibility import for the provider-neutral corpus verifier."""

from cluster.corpus_contract import (  # noqa: F401
    CorpusContractError,
    CorpusEvidence,
    DatasetPointer,
    load_dataset_pointer,
    sha256_file,
    stage_dataset_no_replace,
    verify_canonical_corpus,
    verify_dataset_root,
)

__all__ = [
    "CorpusContractError",
    "CorpusEvidence",
    "DatasetPointer",
    "load_dataset_pointer",
    "sha256_file",
    "stage_dataset_no_replace",
    "verify_canonical_corpus",
    "verify_dataset_root",
]
