"""Independent checks of the Steane purification model (no reuse of qsl.steane's decoder)."""
import itertools, numpy as np, pytest
from qiskit.quantum_info import Pauli
from qsl import steane

H = steane.H
def X(v): return Pauli((np.zeros(7, bool), np.array(v, bool)))
def Z(v): return Pauli((np.array(v, bool), np.zeros(7, bool)))

def test_css_stabilizers_commute_and_logicals():
    stabs = [X(r) for r in H] + [Z(r) for r in H]
    assert all(a.commutes(b) for a in stabs for b in stabs)
    xl, zl = X([1]*7), Z([1]*7)
    assert all(xl.commutes(s) and zl.commutes(s) for s in stabs)
    assert not xl.commutes(zl)

def test_residual_in_normalizer_and_logical_iff_odd_weight():
    zstabs = [Z(r) for r in H]; zl = Z([1]*7)
    for e in itertools.product([0, 1], repeat=7):
        e = np.array(e); s = H @ e % 2
        r = e.copy()
        if s.any(): r[int((s @ [1, 2, 4])) - 1] ^= 1
        assert all(X(r).commutes(z) for z in zstabs)            # residual is a codeword-type error
        assert (not X(r).commutes(zl)) == bool(r.sum() % 2)     # anticommutes with Z_L iff odd weight

def test_matches_independent_min_weight_decoder():
    for q in (0.03, 0.1, 0.25):
        best = {}
        for e in itertools.product([0, 1], repeat=7):
            e = np.array(e); key = tuple(H @ e % 2)
            if key not in best or e.sum() < best[key].sum(): best[key] = e
        tot = 0.0
        for e in itertools.product([0, 1], repeat=7):
            e = np.array(e); r = (e + best[tuple(H @ e % 2)]) % 2
            tot += q**e.sum() * (1-q)**(7-e.sum()) * (r.sum() % 2)
        assert steane.logical_flip_prob(q) == pytest.approx(tot, abs=1e-12)
