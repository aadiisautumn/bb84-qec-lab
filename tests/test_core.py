import numpy as np, pytest
from math import comb
from qsl import qec, bb84, keyrate as K, steane

@pytest.mark.parametrize("code", ["bit3", "phase3", "bit5", "phase5", "shor9"])
def test_single_protected_errors_corrected(code):
    assert qec.verify_single_errors(code) == []

def test_repetition_closed_forms():
    q = 0.2
    nm = qec.NoiseModel(basis_gates=qec.BASIS)
    nm.add_all_qubit_quantum_error(qec.pauli_error([("X", q), ("I", 1 - q)]), ["id"])
    assert qec.run_exact("bit3", "0", nm)[0] == pytest.approx(3*q**2*(1-q) + q**3, abs=1e-9)
    assert qec.run_exact("bit5", "0", nm)[0] == pytest.approx(sum(comb(5, k)*q**k*(1-q)**(5-k) for k in range(3, 6)), abs=1e-9)

def test_unencoded_depolarizing_is_2p_over_3():
    for p in (0.05, 0.3):
        assert qec.basis_averaged("none", qec.pauli_model(p))[0] == pytest.approx(2*p/3, abs=1e-9)

def test_shor_beats_unencoded_at_low_p_only():
    assert qec.basis_averaged("shor9", qec.pauli_model(0.05))[0] < 2*0.05/3
    assert qec.basis_averaged("shor9", qec.pauli_model(0.30))[0] > 2*0.30/3

def test_bb84_noiseless_and_full_intercept():
    assert bb84.run(4000, seed=1)["qber"] == 0
    assert bb84.run(12000, p_eve=1.0, seed=2)["qber"] == pytest.approx(0.25, abs=0.02)

def test_bb84_channel_noise_2p_over_3():
    assert bb84.run(12000, p_noise=0.15, seed=3)["qber"] == pytest.approx(0.10, abs=0.02)

def test_thresholds_and_orderings():
    assert K.Q_TH == pytest.approx(0.110, abs=1e-3)
    d_naive = K.max_distance(lambda L: K.rate_naive(0.5, L), 0, 300)
    d_decoy = K.max_distance(lambda L: K.rate_decoy(0.5, 0.1, L), 0, 300)
    d_gllp = K.max_distance(lambda L: K.best_mu_gllp(L)[1], 0, 200)
    assert d_gllp < d_decoy < d_naive and 160 < d_decoy < 172

def test_steane_pseudo_threshold():
    assert steane.pseudo_threshold() == pytest.approx(0.0969, abs=5e-4)
    assert steane.qber_purified(0.02) < steane.qber_raw(0.02) < steane.qber_purified(0.2)

def test_finite_key_monotone():
    assert K.finite_key_fraction(10**4, 0.03) < K.finite_key_fraction(10**6, 0.03) < K.sp_rate(0.03)
