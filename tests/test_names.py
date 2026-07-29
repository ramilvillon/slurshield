from ngword.names import check_name


def test_obvious_profanity_blocks():
    assert check_name("putangina")["decision"] == "block"

def test_leetspeak_evasion_blocks():
    assert check_name("f00l_you")["reason"] == "fool" or check_name("sh1t")["decision"] == "block"

def test_separator_evasion_blocks():
    # Long slurs (>=5 chars) are caught even with separators (substring on the shadow).
    # Short-slur separator evasion (e.g. "p.u.t.a") is a documented ceiling — not caught.
    assert check_name("p.u.t.a.n.g.i.n.a")["decision"] == "block"

def test_short_slur_as_token_blocks():
    assert check_name("puta")["decision"] == "block"
    assert check_name("xX_puta_Xx")["decision"] == "block"
    assert check_name("puta123")["decision"] == "block"

def test_homoglyph_evasion_blocks():
    # Cyrillic 'а' in place of Latin 'a'
    assert check_name("putа")["decision"] == "block"

def test_scunthorpe_passes():
    assert check_name("scunthorpe")["decision"] == "allow"

def test_assassin_passes():
    assert check_name("assassin123")["decision"] == "allow"

def test_impersonation_blocks():
    assert check_name("admin_official")["reason"] == "impersonation"

def test_impersonation_variants_block():
    for n in ["GM_John", "mod_mike", "staff123", "adminOfficial"]:
        assert check_name(n)["reason"] == "impersonation", n

def test_clean_name_allows():
    r = check_name("shadowhunter")
    assert r["decision"] == "allow" and r["reason"] == "clean"

def test_short_wordlist_tokens_do_not_overblock_names():
    # Regression: 1-2 char / run-collapsed keys ("ho","xx",collapsed "x") must not
    # block ordinary names containing those letters.
    for n in ["Alex", "Max", "Rex", "Dexter", "Phoenix", "Felix", "Roxy", "Jinx"]:
        assert check_name(n)["decision"] == "allow", n

def test_impersonation_substrings_do_not_overblock():
    # Regression: "mod"/"gm"/"staff" as substrings of real words must NOT trip impersonation.
    for n in ["Model", "Modern", "Commodore", "Enigma", "Sigmund", "Gmail_guy", "Staffonly"]:
        assert check_name(n)["decision"] == "allow", n
