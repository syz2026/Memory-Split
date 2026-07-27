"""Natural benchmarks: cloze log-likelihood scoring over HF datasets.

Network-dependent (dataset download); the full suite runs behind `-m slow`
or from scripts. The pure scoring helper loglikelihood_choice_scores is
offline and unit-tested.

Scoring: model.forward(idx[B, T]) -> (logits[B, T, V], None); per choice we
sum log-softmax at the choice-token positions conditioned on the context.
Choice rows are right-padded with EOT — padding sits after every scored
position, so causal attention never sees it.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_GREEDY_MAX_TOKENS = 8  # lambada last-word budget


def loglikelihood_choice_scores(
    model, tok, context: str, choices: list[str], device
) -> list[tuple[float, float]]:
    """Per choice: (sum logprob of choice tokens given context, mean per token).

    One batched forward over all (context + choice) rows. An empty context is
    conditioned on a single EOT (BOS-style), so scoring is always well-defined.
    """
    ctx_ids = tok.encode(context)
    if not ctx_ids:
        ctx_ids = [tok.EOT]
    choice_ids = [tok.encode(c) for c in choices]
    if any(len(c) == 0 for c in choice_ids):
        raise ValueError("every choice must encode to at least one token")

    seqs = [ctx_ids + ch for ch in choice_ids]
    max_len = max(len(s) for s in seqs)
    x = torch.full((len(seqs), max_len), tok.EOT, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        x[i, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)

    with torch.no_grad():
        logits, _ = model.forward(x)
        logprobs = F.log_softmax(logits.float(), dim=-1)

    c = len(ctx_ids)
    out: list[tuple[float, float]] = []
    for i, ch in enumerate(choice_ids):
        # token at position c+j is predicted by logits at position c+j-1
        positions = torch.arange(c - 1, c - 1 + len(ch), device=device)
        targets = torch.tensor(ch, dtype=torch.long, device=device)
        lp = logprobs[i, positions, targets].sum().item()
        out.append((lp, lp / len(ch)))
    return out


def _choice_metrics(model, tok, device, examples, limit) -> dict:
    """examples: iterable of (context, choices, gold_index)."""
    n = n_acc = n_acc_norm = 0
    prob_mass = 0.0
    for context, choices, gold in examples:
        if n >= limit:
            break
        scores = loglikelihood_choice_scores(model, tok, context, choices, device)
        sums = [s for s, _ in scores]
        means = [m for _, m in scores]
        n_acc += int(max(range(len(sums)), key=sums.__getitem__) == gold)
        n_acc_norm += int(max(range(len(means)), key=means.__getitem__) == gold)
        # softmax over per-choice sum logprobs; mass on the gold choice
        mx = max(sums)
        exps = [math.exp(s - mx) for s in sums]
        prob_mass += exps[gold] / sum(exps)
        n += 1
    return {
        "acc": n_acc / n if n else 0.0,
        "acc_norm": n_acc_norm / n if n else 0.0,
        "correct_prob": prob_mass / n if n else 0.0,
        "n": n,
    }


def _greedy_ids(model, tok, ctx_ids: list[int], n_steps: int, device) -> list[int]:
    """Greedy continuation via repeated full forwards (no cache; slow path)."""
    ids = list(ctx_ids)
    out: list[int] = []
    with torch.no_grad():
        for _ in range(n_steps):
            x = torch.tensor([ids], dtype=torch.long, device=device)
            logits, _ = model.forward(x)
            nid = int(logits[0, -1, :].argmax())
            out.append(nid)
            ids.append(nid)
    return out


def _load_dataset(cache_dir, *args, **kwargs):
    from datasets import load_dataset  # lazy: offline unit tests never import it

    # token=False: every suite dataset is public; a stale ambient HF token
    # would otherwise 401 (same pitfall as FineWeb-Edu in build_corpus).
    try:
        return load_dataset(*args, cache_dir=cache_dir, token=False, **kwargs)
    except Exception as e:  # noqa: BLE001 - surface any loader failure clearly
        raise RuntimeError(
            f"could not load HF dataset {args}: the natural suite needs network "
            f"access (or a populated cache_dir). Original error: {e}"
        ) from e


def _iter_hellaswag(ds):
    for ex in ds:
        yield ex["ctx"], [" " + e for e in ex["endings"]], int(ex["label"])


def _iter_arc_easy(ds):
    for ex in ds:
        labels = ex["choices"]["label"]
        if ex["answerKey"] not in labels:
            continue
        gold = labels.index(ex["answerKey"])
        yield ex["question"], [" " + t for t in ex["choices"]["text"]], gold


def _iter_piqa(ds):
    for ex in ds:
        yield ex["goal"], [" " + ex["sol1"], " " + ex["sol2"]], int(ex["label"])


def _iter_winogrande(ds):
    # Choice-in-context: score the full sentence with "_" replaced by each
    # option (empty context => EOT-conditioned), pick the likelier sentence.
    for ex in ds:
        choices = [
            ex["sentence"].replace("_", ex["option1"]),
            ex["sentence"].replace("_", ex["option2"]),
        ]
        yield "", choices, int(ex["answer"]) - 1


def run_natural_suite(
    model,
    tok,
    device,
    tasks=("hellaswag", "arc_easy", "piqa", "winogrande", "lambada"),
    limit: int = 1000,
    cache_dir=None,
) -> dict:
    """Run the natural-benchmark battery; returns {task: metrics dict}.

    Choice tasks report acc (argmax sum logprob), acc_norm (argmax mean
    logprob per token), and correct_prob (mean softmax mass on the gold
    choice). lambada is last-word greedy prediction: correct iff the
    greedy continuation reproduces the gold word's tokens (<= 8 tokens);
    correct_prob is the mean probability of the gold word; acc_norm == acc.
    """
    results: dict[str, dict] = {}
    for task in tasks:
        # namespaced repo ids: datasets>=5 rejects bare canonical names
        if task == "hellaswag":
            ds = _load_dataset(cache_dir, "Rowan/hellaswag", split="validation")
            results[task] = _choice_metrics(model, tok, device, _iter_hellaswag(ds), limit)
        elif task == "arc_easy":
            ds = _load_dataset(cache_dir, "allenai/ai2_arc", "ARC-Easy", split="validation")
            results[task] = _choice_metrics(model, tok, device, _iter_arc_easy(ds), limit)
        elif task == "piqa":
            # ybisk/piqa main branch is a script dataset (unsupported in
            # datasets>=5); the auto-converted parquet branch loads cleanly.
            ds = _load_dataset(cache_dir, "ybisk/piqa", split="validation",
                               revision="refs/convert/parquet")
            results[task] = _choice_metrics(model, tok, device, _iter_piqa(ds), limit)
        elif task == "winogrande":
            ds = _load_dataset(cache_dir, "allenai/winogrande", "winogrande_xl", split="validation")
            results[task] = _choice_metrics(model, tok, device, _iter_winogrande(ds), limit)
        elif task == "lambada":
            # lambada_openai ships a test split only (standard for this eval)
            ds = _load_dataset(cache_dir, "EleutherAI/lambada_openai", split="test")
            results[task] = _lambada_metrics(model, tok, device, ds, limit)
        else:
            raise ValueError(f"unknown natural task: {task!r}")
    return results


def _lambada_metrics(model, tok, device, ds, limit) -> dict:
    n = n_acc = 0
    prob_mass = 0.0
    for ex in ds:
        if n >= limit:
            break
        text = ex["text"].rstrip()
        if " " not in text:
            continue
        context, last_word = text.rsplit(" ", 1)
        target_ids = tok.encode(" " + last_word)
        if not target_ids or len(target_ids) > _GREEDY_MAX_TOKENS:
            continue
        ctx_ids = tok.encode(context)
        pred = _greedy_ids(model, tok, ctx_ids, len(target_ids), device)
        n_acc += int(pred == target_ids)
        (lp, _), = loglikelihood_choice_scores(
            model, tok, context, [" " + last_word], device
        )
        prob_mass += math.exp(lp)
        n += 1
    acc = n_acc / n if n else 0.0
    return {
        "acc": acc,
        "acc_norm": acc,
        "correct_prob": prob_mass / n if n else 0.0,
        "n": n,
    }
