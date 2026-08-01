"""iGSM-in-spirit synthetic math (arXiv 2407.20311), arithmetic mod 23.

Problems are random dependency DAGs over "instance quantities" named by
nonsense (adjective, noun, place) triples, so no world knowledge helps.
Exactly `op` internal definitions are needed to compute the query; the DAG
additionally carries 0-3 distractor definitions that are never used, so the
task requires dependency tracing. The chain-of-thought evaluates needed
nodes in topological order and ends with "\\nAnswer: {n}".

`solve_from_prompt` is an independent oracle: it regex-parses the prompt
text alone and evaluates the graph. It shares only the operator table
(`_OPS`) with the generator, never the generator's value bookkeeping —
this guards against prompt/CoT drift.
"""

from __future__ import annotations

import hashlib
import random
import re
from dataclasses import dataclass

from .records import Doc, QAItem, plain

# Default modulus. It is a difficulty knob, not a constant: the arithmetic
# table has MOD^2 entries per operator, and the 2026-07-20 gates suggest the
# prior floor was a token-budget artifact rather than a MOD artifact. MOD=23
# stays primary; a reduction is a declared fallback with a named trigger, and
# adopting one relabels the construct as "dependency tracing under easy
# modular arithmetic" with no continuity claim to Physics-of-LM numbers.
MOD = 23

# Common words in nonsense combinations ("crimson satchels in the Annex").
ADJECTIVES = (
    "crimson", "amber", "dusty", "velvet", "mossy", "pewter", "cobalt",
    "russet", "sable", "ashen", "briny", "chalky", "dappled", "ochre",
    "umber", "sallow", "tawny", "glossy", "matte", "speckled", "striped",
    "woolly", "wiry", "lumpy", "squat", "slender", "hollow", "gilded",
    "frosted", "smoky", "murky", "pallid", "ruddy", "drab", "garish",
    "mottled", "burnished", "weathered", "crooked", "knotted", "threadbare",
    "brindle",
)

# Singular nouns with clean "+s" plurals.
NOUNS = (
    "satchel", "kite", "flask", "lantern", "spool", "crate", "anvil",
    "ribbon", "marble", "whistle", "bucket", "ladle", "shovel", "plank",
    "barrel", "canteen", "goblet", "trinket", "gadget", "sprocket",
    "pulley", "magnet", "stencil", "easel", "chisel", "mallet", "beaker",
    "funnel", "tripod", "padlock", "thimble", "bobbin", "gasket",
    "grommet", "washer", "dowel", "peg", "cog", "spigot", "valve",
    "hinge", "bracket",
)

PLACES = (
    "Annex", "Loft", "Depot", "Atrium", "Cellar", "Foyer", "Gazebo",
    "Silo", "Wharf", "Pavilion", "Rotunda", "Kiosk", "Vault", "Arcade",
    "Terrace", "Solarium", "Grotto", "Bunker", "Hangar", "Orchard",
    "Quarry", "Mill", "Forge", "Armory", "Aviary", "Apiary", "Greenhouse",
    "Boathouse", "Watchtower", "Courtyard", "Vestibule", "Workshop",
)

# Operator table — the single piece of semantics the oracle may share with
# the generator (plan Task 3). Python % keeps every result in 0..22.
_OPS = {
    "plus": lambda a, b, m: (a + b) % m,
    "minus": lambda a, b, m: (a - b) % m,
    "times": lambda a, b, m: (a * b) % m,
}
_OP_WORDS = tuple(_OPS)


def _regexes(mod: int):
    """Oracle regexes for a given modulus. Cached; the modulus appears in the
    statement text, so a missed hardcode surfaces as an oracle disagreement."""
    if mod not in _RE_CACHE:
        _RE_CACHE[mod] = {
            "leaf": re.compile(r"The number of (.+?) is (\d+)\."),
            "qq": re.compile(
                r"The number of (.+?) equals the number of (.+?) "
                r"(plus|minus|times) the number of (.+?), modulo %d\." % mod
            ),
            "qc": re.compile(
                r"The number of (.+?) equals the number of (.+?) "
                r"(plus|minus|times) (\d+), modulo %d\." % mod
            ),
            "q": re.compile(
                r"Question: What is the number of (.+?), modulo %d\?" % mod
            ),
        }
    return _RE_CACHE[mod]


_RE_CACHE: dict[int, dict] = {}


@dataclass
class IgsmProblem:
    pid: str
    op: int
    prompt: str
    cot: str
    answer: int
    structure_hash: str   # SHA-1 over rendered text: dedup only, NOT structure
    topology_hash: str    # canonical DAG shape: invariant to names/constants
    mod: int


def _fresh_name(rng: random.Random, used: set[tuple[str, str, str]]) -> str:
    while True:
        key = (rng.choice(ADJECTIVES), rng.choice(NOUNS), rng.choice(PLACES))
        if key not in used:
            used.add(key)
            adj, noun, place = key
            return f"{adj} {noun}s in the {place}"


def generate_problem(op: int, rng: random.Random, mod: int = MOD) -> IgsmProblem:
    """Build a DAG whose query needs exactly `op` internal definitions."""
    if op < 1:
        raise ValueError("op must be >= 1")

    used: set[tuple[str, str, str]] = set()
    val: dict[str, int] = {}
    stmt: dict[str, str] = {}
    # (opword, left_operand, right_operand); operand = ("q", name) | ("c", int)
    defn: dict[str, tuple[str, tuple, tuple]] = {}

    def new_leaf() -> str:
        name = _fresh_name(rng, used)
        v = rng.randint(0, mod - 1)
        val[name] = v
        stmt[name] = f"The number of {name} is {v}."
        return name

    def new_internal(left: tuple, opw: str, right: tuple) -> str:
        name = _fresh_name(rng, used)
        a = val[left[1]] if left[0] == "q" else left[1]
        b = val[right[1]] if right[0] == "q" else right[1]
        val[name] = _OPS[opw](a, b, mod)
        if right[0] == "q":
            stmt[name] = (
                f"The number of {name} equals the number of {left[1]} {opw} "
                f"the number of {right[1]}, modulo {mod}."
            )
        else:
            # Constants only ever appear as the right operand.
            stmt[name] = (
                f"The number of {name} equals the number of {left[1]} {opw} "
                f"{right[1]}, modulo {mod}."
            )
        defn[name] = (opw, left, right)
        return name

    def rand_second_operand(prev: str, defined: list[str]) -> tuple:
        r = rng.random()
        if r < 0.35:
            return ("c", rng.randint(1, mod - 1))
        if r < 0.75:
            return ("q", new_leaf())
        return ("q", rng.choice(defined))

    # Needed core: a chain I1..I_op where each internal consumes its
    # predecessor, so every internal is in the query's dependency closure.
    internals: list[str] = []
    first = new_internal(
        ("q", new_leaf()),
        rng.choice(_OP_WORDS),
        ("c", rng.randint(1, mod - 1)) if rng.random() < 0.35 else ("q", new_leaf()),
    )
    internals.append(first)
    for _ in range(op - 1):
        prev = internals[-1]
        defined = list(val)  # all quantities defined so far (all needed)
        other = rand_second_operand(prev, defined)
        opw = rng.choice(_OP_WORDS)
        if other[0] == "c" or rng.random() < 0.5:
            node = new_internal(("q", prev), opw, other)
        else:
            node = new_internal(other, opw, ("q", prev))
        internals.append(node)
    query = internals[-1]
    needed = list(val)  # every quantity so far is in the query's closure

    # Distractor definitions, never needed for the answer. Difficulty floor
    # (2026-07-20 gate-A decision): easy problems carry at most one
    # distractor so dependency tracing is learnable before it is hard.
    max_distractors = 1 if op <= 2 else 3
    for _ in range(rng.randint(0, max_distractors)):
        if rng.random() < 0.5:
            new_leaf()
        else:
            everything = list(val)
            left = ("q", rng.choice(everything))
            r = rng.random()
            if r < 0.35:
                right: tuple = ("c", rng.randint(1, mod - 1))
            else:
                right = ("q", rng.choice(everything))
            new_internal(left, rng.choice(_OP_WORDS), right)

    statements = list(stmt.values())
    rng.shuffle(statements)
    question = f"Question: What is the number of {query}, modulo {mod}?"
    prompt = " ".join(statements) + f"\n{question}\nReasoning:"

    # CoT: needed leaves stated once on first use; each needed internal gets
    # one compute sentence, in construction (= topological) order.
    stated: set[str] = set()
    parts: list[str] = []
    needed_set = set(needed)
    for node in internals:
        opw, left, right = defn[node]
        for operand in (left, right):
            if operand[0] == "q" and operand[1] not in defn:  # a leaf
                leaf = operand[1]
                assert leaf in needed_set
                if leaf not in stated:
                    parts.append(f"The number of {leaf} is {val[leaf]}.")
                    stated.add(leaf)
        a = val[left[1]] if left[0] == "q" else left[1]
        b = val[right[1]] if right[0] == "q" else right[1]
        parts.append(
            f"The number of {node} is ({a} {opw} {b}) mod {mod} = {val[node]}."
        )
    answer = val[query]
    cot = " ".join(parts) + f"\nAnswer: {answer}"

    # Dedup hash over rendered text. It includes entity names drawn from
    # 42*42*32 combinations, so it guarantees exact-text novelty and nothing
    # about structure. Do not describe it as a structural holdout.
    canon = "\n".join(sorted(statements)) + "\n" + question
    structure_hash = hashlib.sha1(canon.encode("utf-8")).hexdigest()

    return IgsmProblem(
        pid=f"igsm-{structure_hash[:12]}",
        op=op,
        prompt=prompt,
        cot=cot,
        answer=answer,
        structure_hash=structure_hash,
        topology_hash=_topology_hash(internals, defn, op),
        mod=mod,
    )


def _topology_hash(internals: list[str], defn: dict, op: int) -> str:
    """Canonical hash of the needed dependency DAG.

    Invariant to entity names, constant values and statement order: operands
    are rewritten as the index of the internal they refer to, or `L` for a
    leaf, or `C` for a constant. Two problems with the same shape and
    different names collide here and not under `structure_hash`, which is
    what makes an out-of-distribution *structure* claim checkable.
    """
    index = {name: i for i, name in enumerate(internals)}

    def slot(operand: tuple) -> str:
        if operand[0] == "c":
            return "C"
        name = operand[1]
        return f"I{index[name]}" if name in index else "L"

    steps = []
    for i, node in enumerate(internals):
        opw, left, right = defn[node]
        steps.append(f"I{i}={slot(left)}:{opw}:{slot(right)}")
    canon = f"op={op}|" + ";".join(steps)
    return hashlib.sha1(canon.encode("utf-8")).hexdigest()


def solve_from_prompt(prompt: str, mod: int = MOD) -> int:
    """Independent oracle: parse the prompt text alone and evaluate mod `mod`.

    Shares only the operator table with the generator, never its value
    bookkeeping. The modulus appears in every statement, so a hardcode the
    parameterisation missed shows up here as an unparseable statement rather
    than as a silently wrong answer.
    """
    rx = _regexes(mod)
    _LEAF_RE, _QQ_RE, _QC_RE, _Q_RE = rx["leaf"], rx["qq"], rx["qc"], rx["q"]
    head = prompt.split("\nQuestion:")[0]
    defs: dict[str, tuple] = {}
    for sent in re.split(r"(?<=\.)\s+", head.strip()):
        sent = sent.strip()
        if not sent:
            continue
        m = _QQ_RE.fullmatch(sent)
        if m:
            defs[m.group(1)] = (m.group(3), ("q", m.group(2)), ("q", m.group(4)))
            continue
        m = _QC_RE.fullmatch(sent)
        if m:
            defs[m.group(1)] = (m.group(3), ("q", m.group(2)), ("c", int(m.group(4))))
            continue
        m = _LEAF_RE.fullmatch(sent)
        if m:
            defs[m.group(1)] = ("leaf", ("c", int(m.group(2))), ("c", 0))
            continue
        raise ValueError(f"unparseable statement: {sent!r}")
    qm = _Q_RE.search(prompt)
    if qm is None:
        raise ValueError("no question found in prompt")

    memo: dict[str, int] = {}

    def ev(name: str) -> int:
        if name in memo:
            return memo[name]
        kind, left, right = defs[name]
        if kind == "leaf":
            result = left[1] % mod
        else:
            a = ev(left[1]) if left[0] == "q" else left[1]
            b = ev(right[1]) if right[0] == "q" else right[1]
            result = _OPS[kind](a, b, mod)
        memo[name] = result
        return result

    return ev(qm.group(1))


def _problem_stream(rng: random.Random, op_lo: int, op_hi: int,
                    low_weighted: bool = False, mod: int = MOD):
    ops = list(range(op_lo, op_hi + 1))
    weights = [1.0 / k for k in ops] if low_weighted else None
    while True:
        op = rng.choices(ops, weights)[0] if weights else rng.randint(op_lo, op_hi)
        yield generate_problem(op, rng, mod=mod)


def generate_igsm_docs(n_docs: int, op_lo: int, op_hi: int, seed: int,
                       low_weighted: bool = True, mod: int = MOD) -> list[Doc]:
    """Training docs weight difficulty toward low op (1/op mass) so the
    easiest levels dominate early learning; eval stays uniform per op."""
    rng = random.Random(seed)
    seen: set[str] = set()
    docs: list[Doc] = []
    if n_docs <= 0:
        return docs
    for p in _problem_stream(rng, op_lo, op_hi, low_weighted=low_weighted, mod=mod):
        if p.structure_hash in seen:
            continue
        seen.add(p.structure_hash)
        full_text = p.prompt + " " + p.cot
        docs.append(
            Doc(
                kind="igsm",
                dense_segments=[plain(full_text)],
                split_segments=[plain(full_text)],
                meta={
                    "structure_hash": p.structure_hash,
                    "topology_hash": p.topology_hash,
                    "op": p.op,
                    "mod": p.mod,
                },
            )
        )
        if len(docs) == n_docs:
            return docs
    return docs


def generate_igsm_eval(
    n_items: int,
    op_lo: int,
    op_hi: int,
    seed: int,
    exclude: set[str],
    mod: int = MOD,
    exclude_topologies: set[str] | None = None,
) -> list[QAItem]:
    """Held-out items. `exclude` is the training text-hash set (dedup);
    `exclude_topologies` additionally holds out DAG shapes, which is what an
    out-of-distribution structure claim needs. An OOD `op` band is disjoint in
    topology by construction, since the chain length is part of the shape."""
    rng = random.Random(seed)
    seen: set[str] = set()
    items: list[QAItem] = []
    if n_items <= 0:
        return items
    for p in _problem_stream(rng, op_lo, op_hi, mod=mod):
        if p.structure_hash in exclude or p.structure_hash in seen:
            continue
        if exclude_topologies and p.topology_hash in exclude_topologies:
            continue
        seen.add(p.structure_hash)
        items.append(
            QAItem(
                qid=f"igsm-{len(items)}",
                task="igsm",
                prompt=p.prompt,
                answer=str(p.answer),
                meta={
                    "op": p.op,
                    "mod": p.mod,
                    "structure_hash": p.structure_hash,
                    "topology_hash": p.topology_hash,
                    "template": f"igsm-op{p.op}",
                    # The worked trace, so the continuous metrics can score
                    # reference-trace NLL. prompt + solution reproduces the
                    # training text byte for byte.
                    "solution": " " + p.cot,
                },
            )
        )
        if len(items) == n_items:
            return items
    return items
