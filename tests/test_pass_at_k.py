from ifstruct_rl.passk import pass_at_k


def test_pass_at_k_n_equals_k():
    assert pass_at_k(8, 0, 8) == 0.0
    assert pass_at_k(8, 1, 8) == 1.0
    assert pass_at_k(8, 8, 8) == 1.0
    assert abs(pass_at_k(8, 2, 1) - 0.25) < 1e-12


def test_pass_at_k_unbiased():
    # 2/10 correct, pass@1 = 0.2; pass@2 = 1 - C(8,2)/C(10,2)
    assert abs(pass_at_k(10, 2, 1) - 0.2) < 1e-12
    expected = 1.0 - (28 / 45)
    assert abs(pass_at_k(10, 2, 2) - expected) < 1e-12
