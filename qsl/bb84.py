"""Gate-level BB84 on Qiskit AerSimulator.

Every transmission is a real circuit execution. Transmissions with identical settings
(Alice bit/basis, Eve intercept basis or none, Bob basis) are the same circuit, so they are
executed as ONE circuit with many shots; each shot is an independent run with independent
channel noise and independent measurement outcomes. Settings are drawn per-transmission
with the RNG and counts are aggregated afterwards.

Eve = intercept-resend: measure in a random basis, re-prepare the measured state (mid-circuit
measure + reset + classically conditioned X/H), exactly as an adversary would.
Channel noise: Pauli (X,Y,Z each p/3) on the 'id' marker between Alice and Bob, so a Z-basis
flip happens with probability 2p/3.
"""
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error


def circuit(bit, a_basis, eve, b_basis):
    """eve in {None, 'Z', 'X'}."""
    qc = QuantumCircuit(1, 2)
    if bit: qc.x(0)
    if a_basis == "X": qc.h(0)
    if eve:                                    # intercept-resend
        if eve == "X": qc.h(0)
        qc.measure(0, 1)
        qc.reset(0)
        with qc.if_test((qc.clbits[1], 1)):
            qc.x(0)
        if eve == "X": qc.h(0)
    qc.id(0)                                   # the channel
    if b_basis == "X": qc.h(0)
    qc.measure(0, 0)
    return qc


def run(n, p_eve=0.0, p_noise=0.0, seed=0, backend=None):
    """Simulate n BB84 transmissions; return dict with sifted length, errors, QBER."""
    rng = np.random.default_rng(seed)
    nm = NoiseModel(basis_gates=["id", "x", "h", "measure", "reset"])
    if p_noise > 0:
        nm.add_all_qubit_quantum_error(pauli_error([("X", p_noise / 3), ("Y", p_noise / 3), ("Z", p_noise / 3), ("I", 1 - p_noise)]), ["id"])
    sim = backend or AerSimulator(noise_model=nm, seed_simulator=int(seed))
    bit = rng.integers(0, 2, n); ab = rng.integers(0, 2, n); bb = rng.integers(0, 2, n)
    intercept = rng.random(n) < p_eve; eb = rng.integers(0, 2, n)
    eve = np.where(intercept, np.where(eb == 0, 1, 2), 0)        # 0 none, 1 Z, 2 X
    keys = bit * 1 + ab * 2 + eve * 4 + bb * 12                   # unique code per setting
    errors = sifted = 0
    for k in np.unique(keys):
        m = int((keys == k).sum())
        b = k % 2; a = (k // 2) % 2; e = (k // 4) % 3; bs = k // 12
        if a != bs:
            continue                                               # discarded in sifting
        qc = circuit(b, "ZX"[a], [None, "Z", "X"][e], "ZX"[bs])
        tc = transpile(qc, sim, optimization_level=0)
        counts = sim.run(tc, shots=m).result().get_counts()
        wrong = sum(v for s, v in counts.items() if int(s[-1]) != b)
        errors += wrong; sifted += m
    return {"n": n, "sifted": sifted, "errors": errors, "qber": errors / max(sifted, 1)}
