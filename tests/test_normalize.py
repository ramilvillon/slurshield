from slurshield.normalize import fold


def test_lowercases_and_strips_zero_width():
    assert fold("F​U​CK") == "fuck"

def test_fullwidth_folds_to_ascii():
    assert fold("ＦＵＣＫ") == "fuck"

def test_leetspeak_folds():
    assert fold("f00l") == "fool"
    assert fold("@ss") == "ass"

def test_separators_removed():
    assert fold("f.u.c.k") == "fuck"
    assert fold("p_u_t_a") == "puta"

def test_long_runs_collapsed():
    assert fold("fuuuuck") == "fuck"

def test_clean_word_unchanged():
    assert fold("hello") == "hello"

def test_cjk_passes_through():
    # NFKC + zero-width strip only; CJK chars survive for substring matching
    assert fold("操​你") == "操你"
