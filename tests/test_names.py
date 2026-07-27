from ngword.names import check_name


def test_obvious_profanity_blocks():
    assert check_name("putangina")["decision"] == "block"

def test_leetspeak_evasion_blocks():
    assert check_name("f00l_you")["reason"] == "fool" or check_name("sh1t")["decision"] == "block"

def test_separator_evasion_blocks():
    assert check_name("p.u.t.a")["decision"] == "block"

def test_homoglyph_evasion_blocks():
    # Cyrillic 'а' in place of Latin 'a'
    assert check_name("putа")["decision"] == "block"

def test_scunthorpe_passes():
    assert check_name("scunthorpe")["decision"] == "allow"

def test_assassin_passes():
    assert check_name("assassin123")["decision"] == "allow"

def test_impersonation_blocks():
    assert check_name("admin_official")["reason"] == "impersonation"

def test_clean_name_allows():
    r = check_name("shadowhunter")
    assert r["decision"] == "allow" and r["reason"] == "clean"
