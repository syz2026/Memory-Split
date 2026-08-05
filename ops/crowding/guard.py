"""The resource ceiling, enforced where the submission happens.

This project does not spend dollars. It spends FarmShare scratch bytes and
L40S GPU-hours, and both have already been overrun once: the account was
cancelled mid-battery at 261 GB against a 150 GB budget that existed only as a
sentence in `ops/crowding/RUNBOOK.md`. A limit written in a runbook is not a
limit. This module is the limit.

Three properties it has to have, each of which the previous arrangement lacked:

    the ceiling comes from the frozen preregistration, not from a constant
    here, so the number the code enforces and the number the design promised
    cannot drift apart. `docs/PREREGISTRATION.md` §10a holds the only copy.

    it binds from the first submission. A guard keyed to "projected total
    against completion fraction" is undefined when nothing has completed, so
    the opening runs sail through unguarded. Every claim here is checked
    against an absolute ceiling with no reference to progress.

    a soft stop holds back the last slice. `may_start_new_work` goes false at
    90% of a campaign ceiling while `claim` keeps honouring work already
    authorised, so in-flight runs drain instead of being truncated. A ceiling
    hit at 95% of a matrix leaves half-finished runs that are unanalysable and
    fully paid for.

Projections use measured throughput only. `d40m_std`'s ~277,000 tok/s is an
extrapolation from `d40m`'s measured 184,671 via an assumed 1.5x FLOPs saving
and has never been observed, so budgeting at it would understate every run by
a third. `MEASURED_TOK_S` is what the guard uses; a faster outturn is headroom
returned, never headroom spent in advance.

CLI, which is what the shell submission paths call:

    python -m ops.crowding.guard claim --campaign fc1 --run fc1_d40m_std_e200 \\
        --tokens 21335900160 --ledger $MS_ROOT/logs/resource_ledger.jsonl

Exit 0 authorises the submission. Exit 3 refuses it and prints why. Any nonzero
exit must abort the submission; `set -e` in the callers does that.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
PREREG = REPO / "docs" / "PREREGISTRATION.md"

# Measured on one L40S at ctx=1024, micro_batch_size=32, compiled, from
# outputs/farmshare-tiny. This is an observation, not a projection.
MEASURED_TOK_S = 184_671

# uint16 target stream plus one uint8 byte per sidecar per token.
BYTES_PER_TOKEN = 4

# fp32 Adam state alongside the weights, per d40m checkpoint.
CHECKPOINT_BYTES = 487_000_000

_CEILING_BLOCK = re.compile(r"```prereg-ceiling\n(.*?)```", re.DOTALL)


class CeilingExceeded(RuntimeError):
    """A submission would breach a preregistered ceiling. Not catchable by
    lowering the ceiling: the value lives in the frozen document."""


class DivergenceHalt(RuntimeError):
    """Measured cost per run exceeded its projection by more than the
    preregistered factor. The cost model is wrong; amend before resuming."""


@dataclass(frozen=True)
class Ceiling:
    scratch_bytes: int
    gpu_hours: float
    soft_stop_fraction: float
    divergence_factor: float
    campaigns: dict[str, float] = field(default_factory=dict)

    @classmethod
    def from_prereg(cls, path: Path | None = None) -> "Ceiling":
        """Parse the frozen ceiling block. Refuses a preregistration that has
        none, rather than falling back to a default -- a default here would be
        an unpreregistered ceiling wearing the same name."""
        path = path or PREREG
        text = path.read_text()
        m = _CEILING_BLOCK.search(text)
        if not m:
            raise CeilingExceeded(
                f"{path} has no ```prereg-ceiling block. The ceiling must come "
                f"from the preregistration; there is no default.")
        vals: dict[str, str] = {}
        for line in m.group(1).splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            k, _, v = line.partition("=")
            vals[k.strip()] = v.strip()
        campaigns = {k.split(":", 1)[1]: float(v)
                     for k, v in vals.items() if k.startswith("campaign:")}
        required = ("scratch_bytes_ceiling", "gpu_hours_ceiling",
                    "soft_stop_fraction", "divergence_factor")
        missing = [k for k in required if k not in vals]
        if missing:
            raise CeilingExceeded(f"prereg ceiling block missing {missing}")
        return cls(
            scratch_bytes=int(vals["scratch_bytes_ceiling"]),
            gpu_hours=float(vals["gpu_hours_ceiling"]),
            soft_stop_fraction=float(vals["soft_stop_fraction"]),
            divergence_factor=float(vals["divergence_factor"]),
            campaigns=campaigns,
        )

    def for_campaign(self, campaign: str) -> float:
        """The binding GPU-hour ceiling for one campaign.

        A campaign the preregistration does not name has no ceiling, and an
        unbounded campaign is the thing this module exists to prevent. So it is
        refused rather than defaulted to the programme total: inventing a new
        campaign name would otherwise be a way to get 1,600 GPU-h without
        writing anything down.
        """
        if campaign not in self.campaigns:
            raise CeilingExceeded(
                f"campaign '{campaign}' has no ceiling in the preregistration. "
                f"Named campaigns are {sorted(self.campaigns)}. Add it to the "
                f"```prereg-ceiling block in docs/PREREGISTRATION.md §10a as an "
                f"amendment before submitting under a new name.")
        return min(self.campaigns[campaign], self.gpu_hours)


def corpus_bytes(total_tokens: int, n_sidecars: int = 2) -> int:
    """Scratch cost of one built load. Mirrors theory.capacity.bytes_on_disk;
    kept here so the guard has no import that could be stubbed out."""
    return total_tokens * (2 + n_sidecars)


def projected_hours(total_tokens: int, tok_s: int = MEASURED_TOK_S) -> float:
    """Wall-clock GPU-hours from a *measured* throughput. Never from a rate
    obtained by multiplying a measurement by an expected speedup."""
    if tok_s <= 0:
        raise ValueError("throughput must be positive")
    return total_tokens / tok_s / 3600.0


class ResourceGuard:
    """Append-only ledger of authorised and completed work, plus the checks.

    One instance per campaign. The ledger is JSONL so a partial write costs at
    most the last line, and so two sessions appending cannot silently
    overwrite each other's accounting.
    """

    def __init__(self, ceiling: Ceiling, ledger_path: Path, campaign: str):
        self.ceiling = ceiling
        self.ledger_path = Path(ledger_path)
        self.campaign = campaign
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)

    def _entries(self) -> list[dict]:
        if not self.ledger_path.exists():
            return []
        out = []
        for line in self.ledger_path.read_text().splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
            
        return out

    def _append(self, rec: dict) -> None:
        with self.ledger_path.open("a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    # ---------------------------------------------------------------- state

    def committed_hours(self, campaign: str | None = None) -> float:
        """Hours already authorised, whether or not they have been spent.

        Claimed-but-unfinished work counts at its projection, because counting
        only finished work would let an unbounded number of runs be authorised
        inside the window before the first one lands. Once a run reports its
        measured cost, that replaces the projection: a ledger that stays
        optimistic after an overrun lets the ceiling drift by exactly the
        amount the cost model was wrong by.
        """
        c = campaign or self.campaign
        entries = self._entries()
        actual = {e["run_id"]: float(e["actual_hours"]) for e in entries
                  if e.get("kind") == "actual" and e.get("campaign") == c}
        total = 0.0
        for e in entries:
            if e.get("kind") != "claim" or e.get("campaign") != c:
                continue
            rid = e.get("run_id")
            total += actual.get(rid, float(e.get("projected_hours", 0.0)))
        return total

    def resident_bytes(self) -> float:
        """Scratch currently held: corpora claimed and not released, plus
        checkpoints for every claimed run."""
        live: dict[str, int] = {}
        ckpt = 0
        for e in self._entries():
            if e.get("kind") == "claim":
                if e.get("corpus"):
                    live[e["corpus"]] = int(e.get("corpus_bytes", 0))
                ckpt += int(e.get("checkpoint_bytes", 0))
            elif e.get("kind") == "release":
                live.pop(e.get("corpus"), None)
                ckpt -= int(e.get("checkpoint_bytes", 0))
        return sum(live.values()) + max(0, ckpt)

    def may_start_new_work(self, campaign: str | None = None) -> bool:
        """Soft stop. False once the campaign is 90% committed, so in-flight
        runs drain rather than being truncated at the hard ceiling."""
        c = campaign or self.campaign
        cap = self.ceiling.for_campaign(c)
        return self.committed_hours(c) < self.ceiling.soft_stop_fraction * cap

    # ---------------------------------------------------------------- claim

    def claim(self, run_id: str, total_tokens: int, *, corpus: str | None = None,
              n_sidecars: int = 2, tok_s: int = MEASURED_TOK_S,
              n_checkpoints: int = 1, dry_run: bool = False) -> dict:
        """Authorise one run, or raise. Call before `sbatch`, never after."""
        hours = projected_hours(total_tokens, tok_s)
        cbytes = corpus_bytes(total_tokens, n_sidecars) if corpus else 0
        ckbytes = CHECKPOINT_BYTES * n_checkpoints
        cap = self.ceiling.for_campaign(self.campaign)

        self._check_divergence()

        committed = self.committed_hours()
        if committed + hours > cap:
            raise CeilingExceeded(
                f"{run_id}: {hours:.2f} GPU-h would take campaign "
                f"'{self.campaign}' to {committed + hours:.2f} against a "
                f"preregistered ceiling of {cap:.0f}. "
                f"Raise it in docs/PREREGISTRATION.md §10a as an amendment, or "
                f"do not submit.")

        if not self.may_start_new_work():
            raise CeilingExceeded(
                f"{run_id}: soft stop. Campaign '{self.campaign}' is "
                f"{committed:.2f}/{cap:.0f} GPU-h, at or past "
                f"{self.ceiling.soft_stop_fraction:.0%}. In-flight work may "
                f"finish; no new work is authorised.")

        projected_scratch = self.resident_bytes() + cbytes + ckbytes
        if projected_scratch > self.ceiling.scratch_bytes:
            raise CeilingExceeded(
                f"{run_id}: projected scratch "
                f"{projected_scratch / 1e9:.1f} GB exceeds the preregistered "
                f"{self.ceiling.scratch_bytes / 1e9:.0f} GB. Release a corpus "
                f"first -- build, train and delete one load at a time.")

        rec = {
            "kind": "claim", "ts": time.time(), "campaign": self.campaign,
            "run_id": run_id, "total_tokens": total_tokens,
            "projected_hours": round(hours, 4), "tok_s": tok_s,
            "corpus": corpus, "corpus_bytes": cbytes,
            "checkpoint_bytes": ckbytes,
        }
        if not dry_run:
            self._append(rec)
        return rec

    def record_actual(self, run_id: str, actual_hours: float) -> dict:
        """Close a claim with its measured cost, and fire the breaker if the
        cost model was wrong."""
        rec = {"kind": "actual", "ts": time.time(), "campaign": self.campaign,
               "run_id": run_id, "actual_hours": float(actual_hours)}
        self._append(rec)
        self._check_divergence()
        return rec

    def release(self, corpus: str, checkpoint_bytes: int = 0) -> None:
        """Give scratch back. Deleting a corpus on disk without recording it
        here leaves the guard refusing submissions it should permit."""
        self._append({"kind": "release", "ts": time.time(),
                      "campaign": self.campaign, "corpus": corpus,
                      "checkpoint_bytes": int(checkpoint_bytes)})

    def _check_divergence(self) -> None:
        claims = {e["run_id"]: e for e in self._entries()
                  if e.get("kind") == "claim"}
        for e in self._entries():
            if e.get("kind") != "actual":
                continue
            c = claims.get(e.get("run_id"))
            if not c:
                continue
            proj = float(c.get("projected_hours", 0.0))
            act = float(e.get("actual_hours", 0.0))
            if proj > 0 and act > proj * self.ceiling.divergence_factor:
                raise DivergenceHalt(
                    f"{e['run_id']}: measured {act:.2f} GPU-h against a "
                    f"projected {proj:.2f}, a factor of {act / proj:.2f} "
                    f"above the preregistered {self.ceiling.divergence_factor}. "
                    f"The cost model of this experiment is wrong. File it as a "
                    f"finding and amend §10a before resuming.")

    def status(self) -> dict:
        cap = self.ceiling.for_campaign(self.campaign)
        committed = self.committed_hours()
        return {
            "campaign": self.campaign,
            "gpu_hours_committed": round(committed, 2),
            "gpu_hours_ceiling": cap,
            "gpu_hours_fraction": round(committed / cap, 4) if cap else None,
            "may_start_new_work": self.may_start_new_work(),
            "resident_scratch_bytes": int(self.resident_bytes()),
            "scratch_ceiling_bytes": self.ceiling.scratch_bytes,
        }


def _default_ledger() -> Path:
    import os
    root = os.environ.get("MS_ROOT")
    base = Path(root) if root else REPO / "outputs"
    return base / "logs" / "resource_ledger.jsonl"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("claim", help="authorise one run, or refuse it")
    p.add_argument("--campaign", required=True)
    p.add_argument("--run", default=None)
    p.add_argument("--tokens", type=int, default=None)
    p.add_argument("--config", default=None,
                   help="run YAML; takes run_id and total_tokens from the same "
                        "file the trainer will read, so the guarded number and "
                        "the submitted number cannot drift apart")
    p.add_argument("--corpus", default=None)
    p.add_argument("--n-sidecars", type=int, default=2)
    p.add_argument("--n-checkpoints", type=int, default=1)
    p.add_argument("--ledger", default=None)
    p.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("status", help="print the ledger position")
    s.add_argument("--campaign", required=True)
    s.add_argument("--ledger", default=None)

    a = sub.add_parser("actual", help="record measured GPU-hours for a run")
    a.add_argument("--campaign", required=True)
    a.add_argument("--run", required=True)
    a.add_argument("--hours", type=float, required=True)
    a.add_argument("--ledger", default=None)

    r = sub.add_parser("release", help="return scratch held by a corpus")
    r.add_argument("--campaign", required=True)
    r.add_argument("--corpus", required=True)
    r.add_argument("--ledger", default=None)

    args = ap.parse_args(argv)
    ledger = Path(args.ledger) if args.ledger else _default_ledger()

    try:
        ceiling = Ceiling.from_prereg()
        guard = ResourceGuard(ceiling, ledger, args.campaign)
        if args.cmd == "claim":
            if args.config:
                import yaml
                cfg = yaml.safe_load(Path(args.config).read_text())
                args.run = args.run or cfg["run_id"]
                args.tokens = args.tokens or int(cfg["total_tokens"])
                if args.corpus is None and cfg.get("train_bin"):
                    args.corpus = Path(cfg["train_bin"]).parent.name
            if not args.run or not args.tokens:
                ap.error("claim needs --config, or both --run and --tokens")
            rec = guard.claim(args.run, args.tokens, corpus=args.corpus,
                              n_sidecars=args.n_sidecars,
                              n_checkpoints=args.n_checkpoints,
                              dry_run=args.dry_run)
            print(f"  AUTHORISED {rec['run_id']}: "
                  f"{rec['projected_hours']:.2f} GPU-h at a measured "
                  f"{rec['tok_s']:,} tok/s, "
                  f"{rec['corpus_bytes'] / 1e9:.1f} GB corpus")
            print(f"  ledger {ledger}")
            print(json.dumps(guard.status(), indent=2))
        elif args.cmd == "status":
            print(json.dumps(guard.status(), indent=2))
        elif args.cmd == "actual":
            guard.record_actual(args.run, args.hours)
            print(json.dumps(guard.status(), indent=2))
        elif args.cmd == "release":
            guard.release(args.corpus)
            print(json.dumps(guard.status(), indent=2))
    except (CeilingExceeded, DivergenceHalt) as exc:
        print(f"\n  REFUSED: {exc}\n", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
