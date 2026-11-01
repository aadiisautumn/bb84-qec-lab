"""Quantum-memory node for BB84: how long can a received qubit be stored before QBER kills the key?

BB84 uses two bases, so a memory must preserve all four cardinal states; the basis-averaged logical
error of qsl.qec IS the memory-induced QBER. It is combined with channel QBER as independent flips.
"""
import numpy as np
from . import qec, keyrate

def e_mem(code, T1_us, T2_us, t_us):
    nm = qec.thermal_model(T1_us, T2_us, t_idle_ns=t_us * 1e3)
    return qec.basis_averaged(code, nm)[0]

def combine(q_chan, e):
    return q_chan + e - 2 * q_chan * e

def curve(code, T1, T2, times):
    return np.array([e_mem(code, T1, T2, t) for t in times])

def crossings(x, a, b):
    """x-values (log-interpolated) where curve a goes below / above curve b."""
    d = a - b; out = []
    for i in range(len(x) - 1):
        if d[i] == 0 or d[i] * d[i + 1] < 0:
            f = d[i] / (d[i] - d[i + 1]) if d[i] != d[i + 1] else 0
            out.append(float(np.exp(np.log(x[i]) + f * (np.log(x[i + 1]) - np.log(x[i])))))
    return out

def t_max(x, e, q_chan=0.0, thr=keyrate.Q_TH):
    """Largest storage time with combined QBER below the Shor-Preskill threshold (log-interp)."""
    tot = combine(q_chan, e)
    idx = np.where(tot >= thr)[0]
    if len(idx) == 0: return float(x[-1])
    i = idx[0]
    if i == 0: return 0.0
    f = (thr - tot[i - 1]) / (tot[i] - tot[i - 1])
    return float(np.exp(np.log(x[i - 1]) + f * (np.log(x[i]) - np.log(x[i - 1]))))
