import hashlib
import os
from pathlib import Path
import subprocess
import sys

from corpusgen.records import lookup_segments, plain
from train.tokenizer import get_tok


REPO_ROOT = Path(__file__).resolve().parents[1]
VENDORED_TIKTOKEN_ASSETS = {
    "6c7ea1a7e38e3a7f062df639a5b80947f075ffe6": (
        "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783"
    ),
    "6d1cbeee0f20b3d9449abfede4726ed8212e3aee": (
        "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5"
    ),
}


def test_gpt2_tiktoken_assets_are_vendored_with_frozen_hashes():
    root = REPO_ROOT / "vendor" / "tiktoken"
    assert {
        path.name for path in root.iterdir() if path.is_file()
    } == set(VENDORED_TIKTOKEN_ASSETS)
    for name, expected in VENDORED_TIKTOKEN_ASSETS.items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected


def test_tokenizer_import_and_encode_work_with_network_blocked():
    script = r"""
import os
from pathlib import Path
import socket

def blocked(*args, **kwargs):
    raise AssertionError("network access attempted")

class OfflineSocket(socket.socket):
    def connect(self, *args, **kwargs):
        blocked(*args, **kwargs)
    def connect_ex(self, *args, **kwargs):
        blocked(*args, **kwargs)

socket.socket = OfflineSocket
socket.create_connection = blocked

from train.tokenizer import get_tok

tok = get_tok()
text = "offline tokenizer <|graph_start|>"
ids = tok.encode(text)
assert tok.decode(ids) == text
cache = Path(os.environ["TIKTOKEN_CACHE_DIR"]).resolve()
assert cache == (Path.cwd() / "vendor" / "tiktoken").resolve()
print("offline-tokenizer-ok")
"""
    env = dict(os.environ)
    env.pop("TIKTOKEN_CACHE_DIR", None)
    env.pop("DATA_GYM_CACHE_DIR", None)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "offline-tokenizer-ok"


def test_special_tokens_atomic():
    tok = get_tok()
    assert tok.encode("<|db_start|>") == [50257]
    assert tok.encode("<|db_retrieve|>") == [50258]
    assert tok.encode("<|db_end|>") == [50259]
    assert tok.encode("<|eot|>") == [50260]


def test_round_trip_plain_text():
    tok = get_tok()
    text = "Kai Nakamura majored in Communications at Stanford University."
    assert tok.decode(tok.encode(text)) == text


def test_encode_segments_mask_alignment():
    tok = get_tok()
    segs = (
        [plain("Kai Nakamura majored in")]
        + lookup_segments("Kai Nakamura", "major", "Communications")
        + [plain(" at Stanford University.")]
    )
    ids, mask = tok.encode_segments(segs, add_eot=True)
    assert len(ids) == len(mask)
    # the masked positions decode to exactly the value segment
    masked_ids = [i for i, m in zip(ids, mask) if m == 0]
    assert tok.decode(masked_ids) == " Communications"
    # full decode reconstructs the concatenated split text + EOT
    full = "".join(t for t, _ in segs) + "<|eot|>"
    assert tok.decode(ids) == full
    # EOT present, loss ON
    assert ids[-1] == tok.EOT and mask[-1] == 1
    # special tokens all receive loss
    for special in (tok.DB_START, tok.DB_RETRIEVE, tok.DB_END):
        positions = [k for k, i in enumerate(ids) if i == special]
        assert positions and all(mask[k] == 1 for k in positions)


def test_encode_segments_no_eot():
    tok = get_tok()
    ids, mask = tok.encode_segments([plain("hello world")], add_eot=False)
    assert tok.EOT not in ids
    assert all(m == 1 for m in mask)


def test_dense_rendering_fully_unmasked():
    tok = get_tok()
    segs = [plain("Kai Nakamura majored in Communications at Stanford University.")]
    ids, mask = tok.encode_segments(segs)
    assert all(m == 1 for m in mask)


def test_graph_special_token_ids_are_reserved_and_atomic():
    tok = get_tok()
    assert tok.GRAPH_START == 50261
    assert tok.GRAPH_MISS == 50275
    assert tok.RELATIONS["r0"] == 50276
    assert tok.RELATIONS["r15"] == 50291
    for text, token_id in tok.graph_special_tokens.items():
        assert tok.encode(text) == [token_id]


def test_paged_identity_delimiters_use_reserved_ids_without_growing_vocab():
    tok = get_tok()

    assert tok.VOCAB_SIZE == 50_304
    assert tok.RELATION_START == 50_292
    assert tok.RELATION_END == 50_293
    assert tok.PAGE_START == 50_294
    assert tok.PAGE_END == 50_295
    for text in (
        "<|relation_start|>",
        "<|relation_end|>",
        "<|page_start|>",
        "<|page_end|>",
    ):
        assert len(tok.encode(text)) == 1
