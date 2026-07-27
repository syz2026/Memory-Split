"""Tests for corpusgen.build (Task 2) — written before the implementation.

corpusgen.igsm_lite / corpusgen.deduction are owned by a parallel task, so
they are stubbed here via sys.modules; build_corpus imports them lazily.
"""

import contextlib
import json
import random
import sys
import types

import numpy as np
import pytest

from corpusgen.build import BuildCfg, build_corpus
from corpusgen.records import Doc, QAItem, plain
from organizer.store import Organizer
from train.tokenizer import get_tok

# ---------------------------------------------------------------- stubs


def _make_stub(kind: str):
    mod = types.ModuleType(f"corpusgen.{'igsm_lite' if kind == 'igsm' else 'deduction'}")

    def gen_docs(n_docs, lo, hi, seed):
        rng = random.Random(seed)
        docs = []
        for i in range(n_docs):
            body = " ".join(
                f"Define q{j} as {rng.randrange(23)} mod 23."
                for j in range(rng.randrange(2, 7))
            )
            text = (
                f"Question: stub {kind} range {lo}-{hi} item {i}.\n"
                f"Reasoning: {body}\nAnswer: {rng.randrange(23)}"
            )
            docs.append(
                Doc(
                    kind=kind,
                    dense_segments=[plain(text)],
                    split_segments=[plain(text)],
                    meta={"structure_hash": f"{kind}-{seed}-{i}"},
                )
            )
        return docs

    def gen_eval(n_items, lo, hi, seed, exclude):
        mod.received_exclude = set(exclude)
        return [
            QAItem(
                qid=f"{kind}-eval-{i}",
                task=kind,
                prompt=f"Question: held-out {kind} item {i}?\nReasoning:",
                answer=str(i % 23),
                meta={"structure_hash": f"{kind}-eval-{seed}-{i}"},
            )
            for i in range(n_items)
        ]

    if kind == "igsm":
        mod.generate_igsm_docs = gen_docs
        mod.generate_igsm_eval = gen_eval
    else:
        mod.generate_deduction_docs = gen_docs
        mod.generate_deduction_eval = gen_eval
    mod.received_exclude = None
    return mod


@contextlib.contextmanager
def reasoning_stubs():
    names = {"corpusgen.igsm_lite": "igsm", "corpusgen.deduction": "deduction"}
    saved = {n: sys.modules.get(n) for n in names}
    stubs = {}
    for module_name, kind in names.items():
        stubs[kind] = _make_stub(kind)
        sys.modules[module_name] = stubs[kind]
    try:
        yield stubs
    finally:
        for module_name, original in saved.items():
            if original is None:
                del sys.modules[module_name]
            else:
                sys.modules[module_name] = original


# ---------------------------------------------------------------- bed fixture

_BED_BASE = [
    "The river cut through the valley long before the first roads were laid.",
    "Glass is made by melting sand together with soda ash and limestone.",
    "Most weather patterns in the region are driven by the westerly winds.",
    "Bees communicate the location of food through a series of movements.",
    "The lighthouse keeper recorded every passing ship in a leather journal.",
    "Copper conducts electricity well and has been used in wiring for a century.",
    "Migrating geese fly in formation to reduce the effort of long flights.",
    "The old mill on the hill ground wheat for the surrounding villages.",
    "Rainfall in the highlands feeds the streams that supply the lowland farms.",
    "Printing presses changed how quickly ideas could travel between cities.",
    "A single oak tree can support hundreds of species of insects and birds.",
    "The harbor freezes in late winter, closing the port for several weeks.",
    "Clay tablets from the ancient city recorded harvests and taxes.",
    "Sound travels faster through water than it does through open air.",
    "The observatory sits above the clouds on the shoulder of the mountain.",
    "Traders once carried salt across the desert in caravans of camels.",
    "Ferns reproduce with spores rather than seeds and prefer damp shade.",
    "The bridge was rebuilt in steel after the flood carried off the timbers.",
    "Volcanic soil is rich in minerals and supports dense orchards.",
    "Early mariners navigated by the stars and the color of the sea.",
    "The library's oldest map shows a coastline that no longer exists.",
    "Wind turbines convert moving air into electricity for the grid.",
    "Cheese ripens in the cellar for months before it reaches the market.",
    "Ants leave chemical trails that guide their nestmates toward food.",
    "The canal connects two rivers and shortens the voyage by a week.",
    "Granite forms slowly underground as molten rock cools over ages.",
    "The orchard blooms in April, drawing beekeepers from three counties.",
    "Telegraph lines once followed the railway across the plains.",
    "Deep lakes hold their summer warmth far into the autumn months.",
    "The potter shapes the clay on a wheel before firing it in a kiln.",
    "Owls hunt at night using hearing precise enough to find mice under snow.",
    "The city's aqueduct still carries water along arches built long ago.",
    "Sailmakers stitched heavy canvas with waxed thread and iron needles.",
    "Moss grows on the north side of trees where sunlight rarely falls.",
    "The train climbs the pass through a spiral of tunnels and bridges.",
    "Glaciers carved these valleys and left behind ridges of gravel.",
    "The market opens at dawn with stalls of fish, bread, and flowers.",
    "Paper was once made from rags beaten to pulp in water mills.",
    "Fireflies signal to one another with short pulses of cold light.",
    "The fort on the headland guarded the strait for three hundred years.",
    "Farmers rotate crops to keep the soil from losing its nutrients.",
    "The choir rehearses in the stone chapel because of its long echo.",
    "Tides in the narrow bay rise faster than a person can walk.",
    "Blacksmiths judged the heat of iron by the color of its glow.",
    "The vineyard terraces follow the curve of the southern slope.",
    "Carrier pigeons delivered messages between the islands for decades.",
    "The dam holds back spring meltwater for the dry months of summer.",
    "Lichen grows a few millimeters a year on the exposed granite.",
    "The ferry crossing takes twenty minutes in calm weather.",
    "Astronomers mapped the comet's path from rooftop observatories.",
]


def bed_fixture():
    def gen():
        i = 0
        while True:
            yield f"{_BED_BASE[i % len(_BED_BASE)]} This is bed passage number {i}."
            i += 1

    return gen()


# ---------------------------------------------------------------- build once

CFG = BuildCfg(
    n_entities=50,
    total_tokens=60_000,
    seed=11,
    n_igsm_eval=25,
    n_deduction_eval=25,
    n_factqa_eval=30,
    n_fresh_entities=20,
    n_fresh_eval=24,
)

TARGETS = CFG.shares()  # whatever the current preregistered mixture is


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("corpus")
    with reasoning_stubs() as stubs:
        report = build_corpus(CFG, get_tok(), bed_fixture(), out)
    return out, report, stubs


# ---------------------------------------------------------------- report


def test_report_shares_within_1pct(built):
    _out, report, _stubs = built
    assert report["checks"]["shares_within_1pct"] is True
    for arm in ("dense", "split"):
        arm_report = report["arms"][arm]
        total = arm_report["total_tokens"]
        assert total == sum(arm_report["component_tokens"].values())
        for comp, target in TARGETS.items():
            share = arm_report["component_tokens"][comp] / total
            assert abs(share - target) <= 0.011, (arm, comp, share)
            assert arm_report["component_shares"][comp] == pytest.approx(share)


def test_report_bio_budgets_and_exposures(built):
    _out, report, _stubs = built
    dense, split = report["arms"]["dense"], report["arms"]["split"]
    # split bios are longer per doc, so the split arm holds fewer exposures
    assert split["component_docs"]["bio"] < dense["component_docs"]["bio"]
    assert split["bio_exposures_per_entity"] < dense["bio_exposures_per_entity"]
    d = dense["component_tokens"]["bio"]
    s = split["component_tokens"]["bio"]
    assert report["bio_cross_arm_rel_diff"] == pytest.approx(abs(s - d) / d)
    assert report["bio_cross_arm_rel_diff"] <= 0.02
    # non-bio synthetic docs are the identical doc lists in both arms
    for comp in ("bed", "igsm", "deduction", "factqa"):
        assert dense["component_docs"][comp] == split["component_docs"][comp]


# ---------------------------------------------------------------- binaries


def test_bins_exist_and_aligned(built):
    out, report, _stubs = built
    for arm in ("dense", "split"):
        bin_path = out / arm / "train.bin"
        mask_path = out / arm / "train.mask.bin"
        assert bin_path.exists() and mask_path.exists()
        ids = np.fromfile(bin_path, dtype=np.uint16)
        mask = np.fromfile(mask_path, dtype=np.uint8)
        assert len(ids) == len(mask) == report["arms"][arm]["total_tokens"]
        assert set(np.unique(mask)) <= {0, 1}


def test_dense_mask_all_ones(built):
    out, _report, _stubs = built
    mask = np.fromfile(out / "dense" / "train.mask.bin", dtype=np.uint8)
    assert (mask == 1).all()


def test_split_mask_zero_fraction(built):
    out, report, _stubs = built
    mask = np.fromfile(out / "split" / "train.mask.bin", dtype=np.uint8)
    frac0 = float((mask == 0).mean())
    assert 0.02 <= frac0 <= 0.25, frac0
    assert report["arms"]["split"]["masked_token_frac"] == pytest.approx(frac0)
    dense_frac = report["arms"]["dense"]["masked_token_frac"]
    assert dense_frac == 0.0


def test_bed_hash_sequence_identical_across_arms(built):
    _out, report, _stubs = built
    assert report["checks"]["bed_identical_across_arms"] is True
    assert (
        report["arms"]["dense"]["bed_hash_digest"]
        == report["arms"]["split"]["bed_hash_digest"]
    )
    assert (
        report["arms"]["dense"]["component_docs"]["bed"]
        == report["arms"]["split"]["component_docs"]["bed"]
    )


# ---------------------------------------------------------------- organizer


def test_organizer_covers_all_entity_relation_pairs(built):
    out, report, _stubs = built
    org = Organizer.load(out / "organizer.jsonl")
    assert len(org) == CFG.n_entities * 6 == 300
    assert report["checks"]["organizer_size_ok"] is True
    # every recall probe must be answerable via the organizer
    probes = [
        json.loads(line)
        for line in (out / "eval" / "recall.jsonl").read_text().splitlines()
    ]
    assert len(probes) == CFG.n_entities * 6
    from corpusgen.bios import RELATION_PHRASES

    for probe in probes:
        relation = probe["meta"]["relation"]
        suffix = f"'s {RELATION_PHRASES[relation]} is"
        assert probe["prompt"].endswith(suffix)
        name = probe["prompt"][: -len(suffix)]
        assert org.lookup(f"{name}, {relation}") == probe["answer"]


# ---------------------------------------------------------------- eval files


def test_eval_files_exist_and_parse(built):
    out, report, stubs = built
    expected = {
        "igsm": CFG.n_igsm_eval,
        "deduction": CFG.n_deduction_eval,
        "factqa": CFG.n_factqa_eval,
        "factqa_fresh": CFG.n_fresh_eval,
        "recall": CFG.n_entities * 6,
    }
    for stem, n in expected.items():
        path = out / "eval" / f"{stem}.jsonl"
        assert path.exists(), stem
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == n, stem
        for row in rows:
            assert set(row) == {"qid", "task", "prompt", "answer", "meta"}
            QAItem(**row)  # constructible
    assert report["eval_counts"] == expected
    # the builder passed train structure hashes for held-out exclusion
    assert stubs["igsm"].received_exclude
    assert stubs["deduction"].received_exclude


# ---------------------------------------------------------------- determinism


def test_parallel_workers_byte_identical(tmp_path):
    """workers>1 must produce byte-identical corpora to workers=1.

    Uses the REAL reasoning generators (not stubs): the worker pool imports
    corpusgen.igsm_lite / corpusgen.deduction in fresh processes, so stubbed
    sys.modules would not propagate anyway.
    """
    import dataclasses

    small = dataclasses.replace(CFG, n_entities=30, total_tokens=30_000,
                                n_igsm_eval=5, n_deduction_eval=5,
                                n_factqa_eval=10, n_fresh_entities=10,
                                n_fresh_eval=5)
    reports = {}
    for label, workers in (("serial", 1), ("parallel", 2)):
        cfg = dataclasses.replace(small, workers=workers)
        reports[label] = build_corpus(cfg, get_tok(), bed_fixture(), tmp_path / label)
    # cfg.workers legitimately differs; everything measured must not
    for r in reports.values():
        r["cfg"].pop("workers")
    assert reports["serial"] == reports["parallel"]
    for arm in ("dense", "split"):
        for fname in ("train.bin", "train.mask.bin"):
            a = (tmp_path / "serial" / arm / fname).read_bytes()
            b = (tmp_path / "parallel" / arm / fname).read_bytes()
            assert a == b, (arm, fname)


def test_rerun_byte_identical(tmp_path):
    reports = []
    for sub in ("a", "b"):
        with reasoning_stubs():
            reports.append(
                build_corpus(CFG, get_tok(), bed_fixture(), tmp_path / sub)
            )
    assert reports[0] == reports[1]
    for arm in ("dense", "split"):
        for fname in ("train.bin", "train.mask.bin"):
            a = (tmp_path / "a" / arm / fname).read_bytes()
            b = (tmp_path / "b" / arm / fname).read_bytes()
            assert a == b, (arm, fname)
    a_org = (tmp_path / "a" / "organizer.jsonl").read_bytes()
    b_org = (tmp_path / "b" / "organizer.jsonl").read_bytes()
    assert a_org == b_org
