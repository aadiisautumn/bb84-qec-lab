"""Repetition codes (3-/5-qubit bit-flip & phase-flip) and Shor's 9-qubit code as explicit
Qiskit circuits, simulated EXACTLY with Aer's density-matrix method (no shot noise).

Decoding = un-encode + syndrome qubits + (perfect, classical) majority correction, which is
equivalent to measure-then-conditionally-correct. Noise is injected through a marker gate 'id'
(storage window) and, in the thermal model, on every gate.

Logical error is averaged over the four cardinal logical inputs |0>,|1>,|+>,|->, because a
repetition code only protects one Pauli type: single-basis testing hides the other half.
"""
import itertools
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error, thermal_relaxation_error

BASIS = ["cx", "h", "x", "z", "s", "sdg", "t", "tdg", "id"]
INPUTS = ["0", "1", "+", "-"]
_G1 = ["h", "x", "z", "s", "sdg", "t", "tdg"]


# ---------------------------------------------------------------- noise models
def pauli_model(p):
    """Depolarizing (Pauli) noise, total error probability p (X,Y,Z each p/3), on the 'id' marker only."""
    nm = NoiseModel(basis_gates=BASIS)
    nm.add_all_qubit_quantum_error(pauli_error([("X", p / 3), ("Y", p / 3), ("Z", p / 3), ("I", 1 - p)]), ["id"])
    return nm


def thermal_model(T1_us, T2_us, t_cx_ns=300, t_1q_ns=50, t_idle_ns=3000):
    """Thermal relaxation (T1/T2) on every gate plus a storage window (t_idle_ns) on 'id'."""
    T1, T2 = T1_us * 1e3, T2_us * 1e3
    if T2 > 2 * T1 + 1e-9:
        raise ValueError("T2 must be <= 2*T1")
    nm = NoiseModel(basis_gates=BASIS)
    e = lambda t: thermal_relaxation_error(T1, T2, t)
    nm.add_all_qubit_quantum_error(e(t_1q_ns), _G1)
    nm.add_all_qubit_quantum_error(e(t_cx_ns).tensor(e(t_cx_ns)), ["cx"])
    nm.add_all_qubit_quantum_error(e(t_idle_ns), ["id"])
    return nm


# ---------------------------------------------------------------- circuits
def _prep(qc, q, inp):
    if inp in ("1", "-"):
        qc.x(q)
    if inp in ("+", "-"):
        qc.h(q)


def _meas_basis(qc, q, inp):
    if inp in ("+", "-"):
        qc.h(q)


def _expected(inp):
    return 1 if inp in ("1", "-") else 0


def build(code, inp, n_idle=1):
    """code in {'none','bit3','phase3','bit5','phase5','shor9'}; returns (circuit, n_data, qubits_to_read)."""
    if code == "none":
        qc = QuantumCircuit(1)
        _prep(qc, 0, inp)
        for _ in range(n_idle):
            qc.id(0)
        _meas_basis(qc, 0, inp)
        return qc, 1, [0]
    if code == "shor9":
        qc = QuantumCircuit(9)
        _prep(qc, 0, inp)
        qc.cx(0, 3); qc.cx(0, 6)
        for q in (0, 3, 6): qc.h(q)
        for b in (0, 3, 6):
            qc.cx(b, b + 1); qc.cx(b, b + 2)
        for _ in range(n_idle):
            for q in range(9): qc.id(q)
        for b in (0, 3, 6):
            qc.cx(b, b + 1); qc.cx(b, b + 2); qc.ccx(b + 1, b + 2, b)
        for q in (0, 3, 6): qc.h(q)
        qc.cx(0, 3); qc.cx(0, 6); qc.ccx(3, 6, 0)
        _meas_basis(qc, 0, inp)
        return qc, 9, [0]
    kind, n = code[:-1], int(code[-1])
    qc = QuantumCircuit(n)
    _prep(qc, 0, inp)
    for i in range(1, n): qc.cx(0, i)
    if kind == "phase":
        for i in range(n): qc.h(i)
    for _ in range(n_idle):
        for i in range(n): qc.id(i)
    if kind == "phase":
        for i in range(n): qc.h(i)
    for i in range(1, n): qc.cx(0, i)          # q_i now holds syndrome bit x0 xor xi
    _meas_basis(qc, 0, inp)
    return qc, n, list(range(n))


def _flip_q0(syn_bits):
    """Majority correction: flip q0 iff more than half of the *other* syndromes... i.e. >= n//2+... see below."""
    n = len(syn_bits) + 1
    return int(sum(syn_bits) >= (n + 1) // 2)  # q0 disagrees with a majority of all n qubits


def logical_error(code, inp, probs, n):
    """Exact probability that decoded logical outcome != expected."""
    exp = _expected(inp)
    if code in ("none", "shor9"):
        return float(probs[1 - exp]) if len(probs) == 2 else None
    err = 0.0
    zin = inp in ("0", "1")
    for idx, pr in enumerate(probs):
        if pr < 1e-15: continue
        bits = [(idx >> i) & 1 for i in range(n)]
        out = bits[0]
        if zin:
            out ^= _flip_q0(bits[1:])
        err += pr * (out != exp)
    return float(err)


# ---------------------------------------------------------------- simulation
_SIM = AerSimulator(method="density_matrix")


def run_exact(code, inp, noise_model, n_idle=1):
    qc, n, rd = build(code, inp, n_idle)
    tc = transpile(qc, basis_gates=BASIS, optimization_level=0)
    ops = dict(tc.count_ops())
    tc.save_probabilities(rd)
    sim = AerSimulator(method="density_matrix", noise_model=noise_model)
    probs = np.array(sim.run(tc, shots=1).result().data()["probabilities"])
    return logical_error(code, inp, probs, n), ops


def basis_averaged(code, noise_model, n_idle=1, inputs=INPUTS):
    vals = [run_exact(code, i, noise_model, n_idle)[0] for i in inputs]
    return float(np.mean(vals)), dict(zip(inputs, vals))


def gate_counts(code):
    qc, _, _ = build(code, "0")
    return dict(transpile(qc, basis_gates=BASIS, optimization_level=0).count_ops())


PROTECTED = {"bit": "X", "phase": "Z", "shor": "XYZ"}


def verify_single_errors(code, paulis=None):
    """Every single-qubit error of the type(s) the code protects, at every position, for all 4
    logical inputs, must be corrected exactly. Returns the list of failures (empty = pass).
    Pass paulis='XYZ' to list *all* failures (repetition codes legitimately fail on the other type)."""
    n = build(code, "0")[1]
    paulis = paulis or PROTECTED[code.rstrip("0123456789")]
    bad = []
    for inp in INPUTS:
        for q, P in itertools.product(range(n), paulis):
            nm = NoiseModel(basis_gates=BASIS)
            nm.add_quantum_error(pauli_error([(P, 1.0)]), ["id"], [q])
            e, _ = run_exact(code, inp, nm, n_idle=1)
            if e > 1e-9:
                bad.append((inp, q, P, e))
    return bad
