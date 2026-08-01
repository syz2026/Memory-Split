#!/usr/bin/env python3
"""Re-check the corpus provenance chain offline, without touching S3.

Every claim in docs/1B-CORPUS-CONSTRUCTION.md that can be checked from bundled
files is checked here: recipe digests against the receipts embedded in the
corpus, receipt digests against the transfer manifest, the manifest against its
own content address, and the shipped producer sources against the generator
digests the extension receipt pins.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

EXPECTED_PREFIX = "84142597cebd96e041d47c7c22dd4b42285b71a213b01265728042cb1a8f6fbb"
CONTRACT_ID = "memorysplit-reasoning-dataset-v3"
BASE_CONTRACT_ID = "memorysplit-parallel-corpus-v2"
TOTAL_TOKENS = 8_169_455_616
BASE_TOKENS = 7_120_879_616
EXTENSION_TOKENS = 1_048_576_000

MANIFEST = "corpus/reasoning-v3-corpus-manifest.json"
POINTER = "corpus/DATASET-POINTER-AWS-135M-V3.json"
FROZEN = "corpus/FROZEN.json"
BASE_RECEIPT = "corpus/receipts/base-receipt.json"
EXT_RECEIPT = "corpus/receipts/extension-receipt.json"
REASONING_POINTER = "corpus/receipts/reasoning-pointer.json"
RECIPE_V2 = "corpus/recipes/reasoning-dataset-v2.json"
RECIPE_V3 = "corpus/recipes/reasoning-dataset-v3.json"
V3_TREE = "build/v3-extension"


class Checker:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.results: list[tuple[bool, str, str]] = []

    def sha(self, relative: str) -> str:
        path = self.root / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def load(self, relative: str) -> dict:
        return json.loads((self.root / relative).read_bytes())

    def check(self, label: str, actual, expected, *, detail: str = "") -> None:
        ok = actual == expected
        if not detail:
            if isinstance(actual, str) and len(actual) == 64:
                detail = actual[:16] if ok else f"{actual[:16]} != {str(expected)[:16]}"
            else:
                detail = str(actual) if ok else f"{actual} != {expected}"
        self.results.append((ok, label, detail))

    @property
    def failures(self) -> int:
        return sum(1 for ok, _, _ in self.results if not ok)

    def report(self) -> None:
        width = max(len(label) for _, label, _ in self.results)
        for ok, label, detail in self.results:
            print(f"  {'ok  ' if ok else 'FAIL'}  {label:<{width}}  {detail}")


def run(root: Path) -> int:
    checker = Checker(root)
    manifest = checker.load(MANIFEST)
    pointer = checker.load(POINTER)
    frozen = checker.load(FROZEN)
    base_receipt = checker.load(BASE_RECEIPT)
    ext_receipt = checker.load(EXT_RECEIPT)
    reasoning_pointer = checker.load(REASONING_POINTER)
    objects = {entry["path"]: entry for entry in manifest["objects"]}

    print("manifest is its own content address")
    manifest_sha = checker.sha(MANIFEST)
    checker.check("manifest sha256 == S3 prefix", manifest_sha, EXPECTED_PREFIX)
    checker.check(
        "pointer records the same manifest",
        pointer["transfer_manifest_sha256"],
        EXPECTED_PREFIX,
    )

    print()
    print("recipes match the receipts embedded in the corpus")
    checker.check(
        "v2 recipe == base receipt source_recipe_sha256",
        checker.sha(RECIPE_V2),
        base_receipt["source_recipe_sha256"],
    )
    recipe_v3 = checker.sha(RECIPE_V3)
    checker.check("v3 recipe == extension receipt recipe_sha256", recipe_v3,
                  ext_receipt["recipe_sha256"])
    checker.check("v3 recipe == reasoning pointer recipe_sha256", recipe_v3,
                  reasoning_pointer["recipe_sha256"])
    checker.check("v3 recipe == FROZEN recipe_sha256", recipe_v3, frozen["recipe_sha256"])

    print()
    print("receipts match the transfer manifest")
    checker.check("base receipt digest", checker.sha(BASE_RECEIPT),
                  objects["base/receipt.json"]["sha256"])
    ext_sha = checker.sha(EXT_RECEIPT)
    checker.check("extension receipt digest", ext_sha,
                  objects["extension/receipt.json"]["sha256"])
    checker.check("extension receipt == manifest virtual_receipt", ext_sha,
                  manifest["virtual_receipt_sha256"])
    checker.check("extension receipt == pointer expected_receipt", ext_sha,
                  reasoning_pointer["expected_receipt_sha256"])
    checker.check("extension receipt == FROZEN receipt_sha256", ext_sha,
                  frozen["receipt_sha256"])
    checker.check("reasoning pointer digest", checker.sha(REASONING_POINTER),
                  objects["locks/reasoning-pointer.json"]["sha256"])
    checker.check("FROZEN digest", checker.sha(FROZEN),
                  objects["locks/FROZEN.json"]["sha256"])
    checker.check("base receipt == v3 recipe base_corpus.receipt_sha256",
                  base_receipt["task4_publication"]["receipt_sha256"],
                  checker.load(RECIPE_V3)["base_corpus"]["receipt_sha256"])

    print()
    print("composite stream digests agree across every record")
    for name in sorted(pointer["streams"]):
        expected = pointer["streams"][name]["sha256"]
        checker.check(f"{name}: manifest", manifest["composite_stream_sha256"][name],
                      expected)
        checker.check(f"{name}: FROZEN", frozen["composite_stream_sha256"][name], expected)
        checker.check(f"{name}: extension receipt",
                      ext_receipt["composite"]["stream_sha256"][name], expected)

    print()
    print("per-object digests agree between manifest and receipts")
    for artifact in base_receipt["artifacts"]:
        entry = objects[f"base/{artifact['path']}"]
        checker.check(f"base/{artifact['path']}", artifact["sha256"], entry["sha256"])
        checker.check(f"base/{artifact['path']} bytes", artifact["bytes"], entry["bytes"])
    for artifact in ext_receipt["extension"]["artifacts"]:
        entry = objects[f"extension/{artifact['path']}"]
        checker.check(f"extension/{artifact['path']}", artifact["sha256"], entry["sha256"])
        checker.check(f"extension/{artifact['path']} bytes", artifact["bytes"],
                      entry["bytes"])

    print()
    print("shipped producer sources match the generator digests in the receipt")
    for relative, expected in sorted(ext_receipt["generator_artifacts"].items()):
        try:
            actual = checker.sha(f"{V3_TREE}/{relative}")
        except FileNotFoundError:
            checker.results.append((False, relative, f"absent from {V3_TREE}/"))
            continue
        checker.check(relative, actual, expected)

    print()
    print("token arithmetic")
    packed = objects["base/packed/targets.bin"]["bytes"]
    ext_packed = objects["extension/packed/targets.bin"]["bytes"]
    checker.check("base tokens (uint16)", packed // 2, BASE_TOKENS)
    checker.check("extension tokens (uint16)", ext_packed // 2, EXTENSION_TOKENS)
    checker.check("composite tokens", (packed + ext_packed) // 2, TOTAL_TOKENS)
    checker.check("manifest raw_target_tokens", manifest["raw_target_tokens"], TOTAL_TOKENS)
    checker.check("pointer raw_target_tokens", pointer["raw_target_tokens"], TOTAL_TOKENS)
    checker.check("FROZEN total_tokens", frozen["total_tokens"], TOTAL_TOKENS)
    checker.check("base sidecar covers base tokens",
                  objects["base/sidecars/dense_target_weights.bin"]["bytes"], BASE_TOKENS)
    checker.check("split90 sidecar covers base tokens",
                  objects["base/sidecars/split90_target_weights.bin"]["bytes"], BASE_TOKENS)
    checker.check("shared sidecar covers extension tokens",
                  objects["extension/sidecars/shared_target_weights.bin"]["bytes"],
                  EXTENSION_TOKENS)
    checker.check("base contract id", base_receipt["contract_id"], BASE_CONTRACT_ID)
    checker.check("composite contract id", manifest["contract_id"], CONTRACT_ID)

    print()
    print("extension task accounting")
    stats = ext_receipt["extension"]["task_stats"]
    checker.check("task count", len(stats), 14)
    checker.check("emitted tokens sum", sum(s["emitted_tokens"] for s in stats),
                  EXTENSION_TOKENS)
    checker.check("emitted records sum", sum(s["emitted_records"] for s in stats),
                  ext_receipt["extension"]["record_count"])
    checker.check("oracle rejections", sum(s["oracle_rejections"] for s in stats), 0)
    checker.check("record count == FROZEN", ext_receipt["extension"]["record_count"],
                  frozen["extension_record_count"])
    checker.check("replayed == emitted", frozen["replayed_records"],
                  frozen["extension_record_count"])

    print()
    checker.report()
    print()
    total = len(checker.results)
    if checker.failures:
        print(f"{checker.failures} of {total} checks FAILED", file=sys.stderr)
        return 1
    print(f"all {total} provenance checks passed")
    return 0


def _find_root(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).resolve()
    here = Path(__file__).resolve()
    for candidate in (here.parent.parent, *here.parents):
        if (candidate / MANIFEST).is_file():
            return candidate
    raise SystemExit(
        f"could not locate a package root containing {MANIFEST}; pass --package-root"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-root", help="root of the unpacked handoff package")
    args = parser.parse_args(argv)
    root = _find_root(args.package_root)
    print(f"package root: {root}\n")
    try:
        return run(root)
    except FileNotFoundError as error:
        print(f"error: required file missing from package: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
