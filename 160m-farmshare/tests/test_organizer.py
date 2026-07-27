from organizer.store import Organizer, normalize


def test_add_lookup():
    org = Organizer()
    org.add("Kai Nakamura", "major", "Communications")
    assert org.lookup("Kai Nakamura, major") == "Communications"
    assert org.hits == 1


def test_normalization():
    org = Organizer()
    org.add("Kai Nakamura", "major", "Communications")
    assert org.lookup("  kai   nakamura,  MAJOR ") == "Communications"
    assert normalize("A  B,   c") == "a b, c"


def test_miss_returns_none_and_counts():
    org = Organizer()
    org.add("Kai Nakamura", "major", "Communications")
    assert org.lookup("Kai Nakamura, employer") is None
    assert org.misses == 1


def test_save_load_round_trip(tmp_path):
    org = Organizer()
    org.add("Kai Nakamura", "major", "Communications")
    org.add("Mia Okafor", "birth_city", "Tacoma")
    path = tmp_path / "org.jsonl"
    org.save(path)
    loaded = Organizer.load(path)
    assert len(loaded) == 2
    assert loaded.lookup("Mia Okafor, birth_city") == "Tacoma"
    assert "Kai Nakamura, major" in loaded


def test_len_and_overwrite():
    org = Organizer()
    org.add("A B", "major", "X")
    org.add("A B", "major", "Y")
    assert len(org) == 1
    assert org.lookup("A B, major") == "Y"
