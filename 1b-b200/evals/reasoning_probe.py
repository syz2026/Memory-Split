"""NR-7 — latent-reasoning probe: readout-vs-learning diagnosis (no retraining).

Answers, from existing checkpoints, *why* reasoning is at chance: does the model
never compute the answer (learning-limited), or compute-but-not-read-out
(readout-limited)? Pure forward passes + a linear probe / logit lens on hidden
states. See replication/specs/nr7-latent-reasoning-probe.md.

Rigor note (the copy trap): the iGSM gold CoT contains the answer verbatim right
before "Answer:", so probing the answer at the post-CoT slot measures COPYING. The
copy-free signal is the answer probed from the END-OF-PROMPT hidden state (answer
not in context there). The post-CoT slot is kept only as an execution/copy control.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from evals.continuous import _split_solution
from evals.mechanism import capture_last_token, fit_linear_probe, probe_accuracy
from evals.scorers import normalize_answer

# ---------------------------------------------------------------- probe sites


def end_of_prompt(item) -> str | None:
    """Context = the raw prompt (ends at 'Reasoning:'); answer NOT in context."""
    return item.prompt


def after_gold_cot(item) -> str | None:
    """Context = prompt + gold CoT through 'Answer:' (copy control; needs solution)."""
    sol = item.meta.get("solution")
    if not sol:
        return None
    split = _split_solution(item.prompt, sol)  # (ctx_incl_answer_tag, answer_cont)
    return split[0] if split else None


# ---------------------------------------------------------------- labels


def _answer_classes(items) -> dict[str, int]:
    vocab = sorted({normalize_answer(it.answer) for it in items})
    return {a: i for i, a in enumerate(vocab)}


def _gold_first_token(tok, answer: str) -> int:
    ids = tok.encode(" " + answer)
    return ids[0] if ids else tok.EOT


def _split_train_test(n: int, test_frac: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_test = max(1, int(n * test_frac))
    test = torch.zeros(n, dtype=torch.bool)
    test[perm[:n_test]] = True
    return ~test, test


# ---------------------------------------------------------------- core


def answer_class_probe(
    model, tok, items, position_fn, device,
    batch_size: int = 16, test_frac: float = 0.3, seed: int = 0,
) -> dict | None:
    """Per-layer linear-probe accuracy for the answer class at one probe site.

    Captures the last-token residual of ``position_fn(item)`` per layer, fits a
    held-out linear probe (answer class), returns
    ``{acc_by_layer, chance, best_layer, best_acc, n_test, decodable}`` or None if
    no item yields a context (e.g. after_gold_cot without solutions).
    """
    pairs = [(it, position_fn(it)) for it in items]
    pairs = [(it, ctx) for it, ctx in pairs if ctx]
    if len(pairs) < 4:
        return None
    used_items = [it for it, _ in pairs]
    contexts = [ctx for _, ctx in pairs]

    classes = _answer_classes(used_items)
    n_classes = len(classes)
    if n_classes < 2:
        return None
    labels = torch.tensor([classes[normalize_answer(it.answer)] for it in used_items])

    resid = capture_last_token(model, tok, contexts, device, batch_size)["resid"]  # [N,L,D]
    N, L, _ = resid.shape
    tr, te = _split_train_test(N, test_frac, seed)
    if int(tr.sum()) < 2 or int(te.sum()) < 1:
        return None

    acc_by_layer = {}
    for layer in range(L):
        X = resid[:, layer, :]
        probe = fit_linear_probe(X[tr], labels[tr], n_classes, seed=seed)
        acc_by_layer[layer] = probe_accuracy(probe, X[te], labels[te])

    best_layer = max(acc_by_layer, key=acc_by_layer.get)
    best_acc = acc_by_layer[best_layer]
    chance = 1.0 / n_classes
    n_test = int(te.sum())
    se = math.sqrt(max(best_acc * (1 - best_acc), 1e-9) / n_test)
    return {
        "acc_by_layer": acc_by_layer,
        "chance": chance,
        "n_classes": n_classes,
        "best_layer": int(best_layer),
        "best_acc": float(best_acc),
        "n_test": n_test,
        "decodable": bool(best_acc - chance > max(0.05, 2 * se)),
    }


def answer_logit_lens(
    model, tok, items, position_fn, device, batch_size: int = 16,
) -> dict | None:
    """Per-layer logit-lens readout of the gold answer's first token at one site.

    Applies the model's own ``lm_head(ln_f(·))`` to each block's residual and
    reports mean p(gold first token) and top-1 hit rate per layer. No fitting;
    deterministic. Returns ``{gold_prob_by_layer, top1_by_layer, n}`` or None.
    """
    pairs = [(it, position_fn(it)) for it in items]
    pairs = [(it, ctx) for it, ctx in pairs if ctx]
    if not pairs:
        return None
    used_items = [it for it, _ in pairs]
    contexts = [ctx for _, ctx in pairs]
    gold = torch.tensor([_gold_first_token(tok, it.answer) for it in used_items])

    resid = capture_last_token(model, tok, contexts, device, batch_size)["resid"]  # [N,L,D]
    N, L, _ = resid.shape
    ln_f, head = model.ln_f, model.lm_head
    gold_prob, top1 = {}, {}
    for layer in range(L):
        h = resid[:, layer, :].to(device)
        with torch.no_grad():
            logits = head(ln_f(h)).float()  # [N, V]
            logp = F.log_softmax(logits, dim=-1)
        idx = torch.arange(N)
        gold_prob[layer] = float(logp[idx, gold].exp().mean())
        top1[layer] = float((logits.argmax(-1).cpu() == gold).float().mean())
    return {"gold_prob_by_layer": gold_prob, "top1_by_layer": top1, "n": N}


# ---------------------------------------------------------------- orchestrator


def _verdict(eop: dict | None, cot: dict | None, greedy_acc: float | None) -> str:
    e = eop and eop["decodable"]
    c = cot and cot["decodable"]
    if e:
        return ("readout-limited: the answer is linearly decodable from the "
                "end-of-prompt state (latent one-pass computation) yet greedy "
                "output is at chance — the model computes but does not verbalize it")
    if c:
        return ("CoT-generation-limited: the answer is decodable only given the "
                "gold CoT — the model can execute correct reasoning but not "
                "produce it (planning/search failure)")
    return ("execution/learning-limited: the answer is not decodable even given "
            "the gold CoT — the final step itself is not learned (P0 ladder)")


def reasoning_readout_report(
    model, tok, items, device,
    greedy_acc: float | None = None, batch_size: int = 16,
    test_frac: float = 0.3, seed: int = 0,
) -> dict:
    """Full readout diagnosis for one arm/task: probe + logit-lens at both sites,
    the greedy-accuracy contrast, and a pre-stated verdict (NR-7 spec 4)."""
    eop = answer_class_probe(model, tok, items, end_of_prompt, device,
                             batch_size, test_frac, seed)
    cot = answer_class_probe(model, tok, items, after_gold_cot, device,
                             batch_size, test_frac, seed)
    report = {
        "n_items": len(items),
        "greedy_acc": greedy_acc,
        "probe_end_of_prompt": eop,
        "probe_after_gold_cot": cot,
        "lens_end_of_prompt": answer_logit_lens(model, tok, items, end_of_prompt, device, batch_size),
        "lens_after_gold_cot": answer_logit_lens(model, tok, items, after_gold_cot, device, batch_size),
        "verdict": _verdict(eop, cot, greedy_acc),
    }
    if eop and greedy_acc is not None:
        report["readout_gap_end_of_prompt"] = eop["best_acc"] - greedy_acc
    return report
