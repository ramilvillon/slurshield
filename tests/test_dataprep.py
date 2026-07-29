from ngword.dataprep import stratified_split, _toxicn_class


def test_toxicn_toxic_first_collapse():
    # clean only when toxic==0; the 816 toxic-but-untyped (expression=0) rows -> explicit
    assert _toxicn_class(0, 0) == 0
    assert _toxicn_class(1, 0) == 1
    assert _toxicn_class(1, 1) == 1
    assert _toxicn_class(1, 2) == 2
    assert _toxicn_class(1, 3) == 3
    # pre-collapsed file (no `toxic` column): expression passes through unchanged
    assert _toxicn_class(None, 2) == 2
    assert _toxicn_class(None, 0) == 0


def _rows():
    rows = []
    for lang in ("en", "zh", "tl", "ja"):
        for label in (0, 1):
            for i in range(50):
                rows.append({"text": f"{lang}-{label}-{i}", "label": label, "lang": lang, "source": "t"})
    return rows


def test_split_ratios_and_no_leakage():
    train, val, test = stratified_split(_rows(), seed=42)
    total = len(train) + len(val) + len(test)
    assert total == 400
    # ~80/10/10
    assert 0.75 < len(train) / total < 0.85
    # no text appears in more than one split
    texts_train = {r["text"] for r in train}
    texts_val = {r["text"] for r in val}
    texts_test = {r["text"] for r in test}
    assert not (texts_train & texts_val)
    assert not (texts_train & texts_test)
    assert not (texts_val & texts_test)

def test_split_is_deterministic():
    a = stratified_split(_rows(), seed=42)[0]
    b = stratified_split(_rows(), seed=42)[0]
    assert [r["text"] for r in a] == [r["text"] for r in b]
