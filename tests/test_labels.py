from ngword.labels import harmonize, dedup


def test_conda_direct_mapping():
    assert harmonize("conda", "O") == 0
    assert harmonize("conda", "E") == 1
    assert harmonize("conda", "I") == 2
    assert harmonize("conda", "A") == 3

def test_binary_sources_map_toxic_to_explicit():
    assert harmonize("mginoben", 0) == 0
    assert harmonize("mginoben", 1) == 1
    assert harmonize("cold", 1) == 1
    assert harmonize("textdetox", "toxic") == 1
    assert harmonize("textdetox", "neutral") == 0

def test_toxicn_keeps_implicit():
    assert harmonize("toxicn", "implicit") == 2
    assert harmonize("toxicn", "explicit") == 1
    assert harmonize("toxicn", "non") == 0

def test_unknown_label_drops_row():
    assert harmonize("cold", 99) is None

def test_dedup_removes_case_and_space_duplicates():
    rows = [
        {"text": "GG ez", "label": 1},
        {"text": "gg  ez", "label": 1},
        {"text": "different", "label": 0},
    ]
    assert len(dedup(rows)) == 2
