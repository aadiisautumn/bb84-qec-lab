"""Steane [[7,1,3]] CSS code used as one-sided EPR purifier (the Shor-Preskill picture).

Noise: i.i.d. depolarizing on Bob's 7 qubits (X,Y,Z each p/3). CSS => X- and Z-type errors are
decoded independently with the [7,4] Hamming code. The key-bit error of the purified pair in the
Z (X) basis is the probability that the residual X (Z) pattern is a logical operator (odd weight).
Because errors on different qubits are independent, this is an EXACT enumeration over 2^7 patterns
(no sampling). Unpurified QBER is 2p/3.
"""
import itertools
import numpy as np

H = np.array([[(j >> b) & 1 for j in range(1, 8)] for b in range(3)])   # column j = binary(j)

def logical_flip_prob(q):
    """P(residual error pattern has odd weight) for i.i.d. flip prob q per qubit."""
    tot = 0.0
    for e in itertools.product([0, 1], repeat=7):
        e = np.array(e); w = e.sum()
        pr = q**w * (1 - q)**(7 - w)
        s = int((H @ e % 2) @ np.array([1, 2, 4]))
        r = e.copy()
        if s: r[s - 1] ^= 1
        tot += pr * (r.sum() % 2)
    return tot

def qber_purified(p): return logical_flip_prob(2 * p / 3)
def qber_raw(p): return 2 * p / 3

def pseudo_threshold():
    from scipy.optimize import brentq
    return brentq(lambda p: qber_purified(p) - qber_raw(p), 0.03, 0.4)
