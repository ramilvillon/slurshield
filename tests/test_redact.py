import pathlib

from slurshield.infer import redact
from slurshield.names import profanity_spans
from slurshield.normalize import fold, fold_map

_WORDLISTS = pathlib.Path(__file__).resolve().parents[1] / "wordlists"


def _corpus() -> list[str]:
    out = []
    for f in sorted(_WORDLISTS.glob("*.txt")):
        out += [ln.strip() for ln in f.read_text(encoding="utf-8").splitlines()
                if ln.strip() and not ln.startswith("#")]
    return out


def test_fold_map_shadow_matches_fold():
    # fold() is now fold_map()[0]; this pins the two together over the real wordlists
    # plus the evasion shapes that made folding lossy in the first place.
    for s in _corpus() + ["f.u.c.k", "shiiiit", "ＦＵＣＫ off", "f  u  c  k", "sh1t",
                          "xX_puta_Xx", "操你妈", "", "   ", "éclair", "ﬁsh"]:
        assert fold_map(s)[0] == fold(s), s


def test_fold_map_spans_stay_inside_the_source():
    for s in _corpus() + ["f  u  c  k", "shiiiit", "ＦＵＣＫ"]:
        shadow, src = fold_map(s)
        assert len(shadow) == len(src)
        assert all(0 <= a < b <= len(s) for a, b in src), s


def test_redact_masks_plain_and_obfuscated():
    assert redact("you are sh1t") == "you are ****"
    # Separator evasion is caught for long slurs only (the tier ceiling in names.py).
    assert redact("p.u.t.a.n.g.i.n.a stop") == "*" * 17 + " stop"
    assert redact("xX_puta_Xx") == "xX_****_Xx"


def test_redact_covers_whole_collapsed_run():
    # 'shiiiit' folds to 'shit' (7 -> 4); the span must still cover all 7 chars.
    masked = redact("shiiiit")
    assert masked == "*" * 7, masked


def test_redact_leaves_clean_text_alone():
    for s in ["class starts in 5", "assemble at mid", "i need to pass", "Scunthorpe", "Phoenix"]:
        assert redact(s) == s, s


def test_redact_leaves_model_only_toxicity_alone():
    # No word-list hit -> nothing to mask, even though classify() blocks this as `explicit`.
    assert redact("you are absolute trash") == "you are absolute trash"


def test_redact_preserves_length_and_accepts_non_str():
    s = "f.u.c.k off sh1t"
    assert len(redact(s)) == len(s)
    assert redact(None) == ""


def test_profanity_spans_are_ordered_and_cover_the_term():
    # Wordlist entries overlap ("putangina" contains "tangina"), so spans may too —
    # redact() masks by position, which makes overlap harmless.
    text = "sh1t and putangina"
    spans = profanity_spans(text)
    assert [s[0] for s in spans] == sorted(s[0] for s in spans)
    assert text[spans[0][0]:spans[0][1]] == "sh1t"
    assert text[spans[1][0]:spans[1][1]] == "putangina"
    assert redact(text) == "**** and *********"
