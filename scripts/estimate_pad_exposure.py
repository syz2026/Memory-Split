"""How much of each evaluation the padding defect could actually have touched.

The defect (docs/RESULTS-2026-08-04.md) left-padded every prompt to the batch
maximum with EOT and attended over the pads. Corruption is monotone in pad
length and sets in between 16 and 24 pads. Pad depth for an item is therefore
    (longest prompt in its batch) - (its own prompt),
which depends only on the prompt-length distribution and the batch size. No
checkpoint and no GPU are needed to compute it, so this runs while the cluster
is unreachable.

The point is triage. An evaluation whose prompts are all nearly the same length
was barely padded and its numbers may survive re-measurement; one with a wide
length spread was hit hard and has to be re-run.

Usage:  PYTHONPATH=. python3 scripts/estimate_pad_exposure.py
"""
from __future__ import annotations

import statistics
from pathlib import Path

from corpusgen import bios, deduction, igsm_lite
from corpusgen.bios import RELATION_PHRASES
from train.tokenizer import get_tok

ROOT = Path(__file__).resolve().parents[1]

# The 0.8B gate runs used the evaluation driver's default batch size of 16.
BATCH_SIZE = 16
# docs/PAPER-MEASUREMENT.md section 3: exact at 0-16 pads, broken by 24.
SAFE_PADS, BROKEN_PADS = 16, 24


def pad_depths(lengths: list[int], batch_size: int) -> list[int]:
    """Pad depth per item under left-pad-to-batch-maximum, in arrival order."""
    out: list[int] = []
    for i in range(0, len(lengths), batch_size):
        chunk = lengths[i : i + batch_size]
        top = max(chunk)
        out.extend(top - n for n in chunk)
    return out


def report(name: str, prompts: list[str], tok) -> None:
    lengths = [len(tok.encode(p)) for p in prompts]
    pads = pad_depths(lengths, BATCH_SIZE)
    n = len(pads)
    safe = sum(1 for p in pads if p <= SAFE_PADS) / n
    broken = sum(1 for p in pads if p >= BROKEN_PADS) / n
    print(f"\n{name}  (n={n}, batch={BATCH_SIZE})")
    print(f"  prompt tokens   min {min(lengths)}  median "
          f"{int(statistics.median(lengths))}  max {max(lengths)}  "
          f"spread {max(lengths) - min(lengths)}")
    print(f"  pad depth       mean {statistics.mean(pads):.1f}  "
          f"median {int(statistics.median(pads))}  max {max(pads)}")
    print(f"  within safe (<={SAFE_PADS} pads)   {safe:6.1%}")
    print(f"  past broken (>={BROKEN_PADS} pads) {broken:6.1%}")
    verdict = ("LIKELY INTACT" if broken < 0.05 else
               "PARTLY CORRUPTED" if broken < 0.50 else "HEAVILY CORRUPTED")
    print(f"  verdict         {verdict}")


def main() -> int:
    tok = get_tok()

    print("=" * 74)
    print("Pad exposure under the pre-fix batching, from prompt lengths alone")
    print("=" * 74)

    # Fact-use QA over fresh entities: one fixed template, only a name and a
    # relation phrase vary, so lengths should be tightly clustered.
    n_entities, n_fresh = 200_000, 200
    recs = bios.generate_records(n_entities + n_fresh, 0)
    fresh = recs[n_entities:]
    relations = list(RELATION_PHRASES)
    fq = [
        f"Question: What is {r.name}'s {RELATION_PHRASES[relations[i % len(relations)]]}?"
        "\nReasoning:"
        for i, r in enumerate(fresh)
    ]
    report("factqa_fresh  (the mechanism result)", fq, tok)

    # iGSM: prompt length grows with operation count, which is exactly the axis
    # the defect rides.
    ig = igsm_lite.generate_igsm_eval(1500, 1, 4, 10_000_000, set())
    report("igsm  (op 1-4)", [it.prompt for it in ig], tok)
    for op in (1, 2, 3, 4):
        sel = [len(tok.encode(it.prompt)) for it in ig if it.meta.get("op") == op]
        if sel:
            print(f"    op{op}: mean prompt {statistics.mean(sel):6.1f} tokens (n={len(sel)})")

    # Deduction: a rule list of variable length.
    ded = deduction.generate_deduction_eval(1500, 1, 2, 30_000_000, set())
    report("deduction", [it.prompt for it in ded], tok)

    # The 1B held-out set carries its own token counts, so no reconstruction is
    # needed. Its scorer is teacher-forced and was never committed, so whether
    # it padded at all is unknown; this is the exposure *if* it batched the way
    # the generative path did. Scoring was per family, so batches are within-family.
    import json

    items: dict[str, list[int]] = {}
    with open(ROOT / "outputs" / "1b-heldout" / "items.jsonl") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            items.setdefault(r["task"], []).append(
                int(r["prompt_tokens"]) + int(r["answer_tokens"])
            )

    print("\n" + "=" * 74)
    print("1B held-out Reasoning-Gym, per family (batches are within-family)")
    print("Upper bound: the teacher-forced scorer was never committed, so we")
    print("cannot confirm it padded. This is the exposure if it padded as the")
    print("generative path did.")
    print("=" * 74)
    print(f"{'family':<22}{'spread':>8}{'mean pad':>10}{'>=24 pads':>11}   verdict")
    appendix = {"rotate_matrix", "binary_matrix", "futoshiki", "path_star", "spiral_matrix"}
    for task in sorted(items):
        lengths = items[task]
        pads = pad_depths(lengths, BATCH_SIZE)
        broken = sum(1 for p in pads if p >= BROKEN_PADS) / len(pads)
        verdict = ("LIKELY INTACT" if broken < 0.05 else
                   "PARTLY CORRUPTED" if broken < 0.50 else "HEAVILY CORRUPTED")
        star = " *" if task in appendix else ""
        print(f"{task + star:<22}{max(lengths) - min(lengths):>8}"
              f"{statistics.mean(pads):>10.1f}{broken:>10.1%}   {verdict}")
    print("  * appears in the appendix table sent to the manager")

    print("\n" + "=" * 74)
    print("Pad depth is (batch maximum - own length), so it is a property of the")
    print("prompt mix and the batch size, not of the model. These figures bound")
    print("which committed numbers can survive re-measurement.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
