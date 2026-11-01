#!/usr/bin/env python3
"""Reproduce every number and figure.  python run_all.py [--quick]"""
import argparse, json, os, time
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qsl import bb84, qec, keyrate as K, steane, memory

ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
R = {}; T0 = time.time()
def log(m): print(f"[{time.time()-T0:6.1f}s] {m}", flush=True)
def save(name): plt.tight_layout(); plt.savefig(f"figures/{name}.png", dpi=150); plt.close()

# ---- E0 circuit self-checks ------------------------------------------------------------
R["verify"] = {c: len(qec.verify_single_errors(c)) for c in ["bit3", "phase3", "bit5", "phase5", "shor9"]}
log(f"single-error verification failures (must be all 0): {R['verify']}")
R["gate_counts"] = {c: qec.gate_counts(c) for c in ["bit3", "bit5", "shor9"]}

# ---- E1 BB84 gate level ----------------------------------------------------------------
N = 4000 if a.quick else 20000
pe = np.linspace(0, 1, 11); r = [bb84.run(N, p_eve=p, seed=i) for i, p in enumerate(pe)]
R["bb84_eve"] = [{"p_eve": float(p), **x} for p, x in zip(pe, r)]
pn = np.linspace(0, 0.5, 11); r2 = [bb84.run(N, p_noise=p, seed=100 + i) for i, p in enumerate(pn)]
R["bb84_noise"] = [{"p": float(p), **x} for p, x in zip(pn, r2)]
se = lambda q, n: float(np.sqrt(q * (1 - q) / max(n, 1)))
plt.figure(figsize=(9, 3.4))
plt.subplot(1, 2, 1)
plt.errorbar(pe, [x["qber"] for x in r], [se(x["qber"], x["sifted"]) * 1.96 for x in r], fmt="o-", label="gate-level (Aer)")
plt.plot(pe, pe / 4, "r--", label=r"theory $p_{eve}/4$"); plt.axhline(K.Q_TH, ls=":", c="gray", label="11.0% threshold")
plt.xlabel(r"interception probability $p_{eve}$"); plt.ylabel("QBER"); plt.legend(fontsize=7); plt.title("Intercept-resend")
plt.subplot(1, 2, 2)
plt.errorbar(pn, [x["qber"] for x in r2], [se(x["qber"], x["sifted"]) * 1.96 for x in r2], fmt="s-", c="g", label="gate-level (Aer)")
plt.plot(pn, 2 * pn / 3, "r--", label="theory 2p/3"); plt.axhline(K.Q_TH, ls=":", c="gray")
plt.xlabel("Pauli channel error p"); plt.legend(fontsize=7); plt.title("Channel noise only")
save("fig1_bb84"); log("E1 BB84 done")

# ---- E2 depolarizing code-capacity ------------------------------------------------------
ps = np.array([0.01, 0.02, 0.05, 0.08, 0.1, 0.12, 0.15, 0.2, 0.3, 0.4, 0.5])
codes = ["none", "bit3", "phase3", "bit5", "shor9"]
E2 = {c: [qec.basis_averaged(c, qec.pauli_model(p))[0] for p in ps] for c in codes}
R["depol"] = {"p": ps.tolist(), **{c: v for c, v in E2.items()}}
fine = np.linspace(0.05, 0.2, 31)
sh = np.array([qec.basis_averaged("shor9", qec.pauli_model(p))[0] for p in fine]) - 2 * fine / 3
R["shor_breakeven_p"] = float(np.interp(0, sh, fine)) if sh[0] < 0 < sh[-1] else None
# closed forms (pure-X noise, protected input only)
from math import comb
q = 0.2; nm = qec.NoiseModel(basis_gates=qec.BASIS); nm.add_all_qubit_quantum_error(qec.pauli_error([("X", q), ("I", 1 - q)]), ["id"])
R["closed_form"] = {"bit3_exact": qec.run_exact("bit3", "0", nm)[0], "bit3_formula": 3 * q**2 * (1 - q) + q**3,
                    "bit5_exact": qec.run_exact("bit5", "0", nm)[0],
                    "bit5_formula": sum(comb(5, k) * q**k * (1 - q)**(5 - k) for k in range(3, 6))}
plt.figure(figsize=(5.2, 3.6))
for c, st in zip(codes, ["k--", "C0o-", "C1s-", "C2^-", "C3d-"]): plt.plot(ps, E2[c], st, label=c, ms=4)
plt.xlabel("depolarizing error p per qubit (storage)"); plt.ylabel("basis-averaged logical error"); plt.legend(fontsize=7)
plt.title("Code-capacity, exact density matrix"); save("fig2_depol"); log(f"E2 done, Shor break-even p = {R['shor_breakeven_p']}")

# ---- E3 thermal T2 sweep (T1 = 80 us, 3 us storage) -------------------------------------
T2s = np.array([8, 12, 16, 24, 30, 40, 50, 65, 79, 95, 110, 135, 160], float)
c5 = ["none", "bit3", "phase3", "bit5", "phase5"]
E3 = {c: [qec.basis_averaged(c, qec.thermal_model(80, t2))[0] for t2 in T2s] for c in c5}
R["thermal_T2"] = {"T2": T2s.tolist(), **E3}
R["bit_phase_crossover_T2"] = memory.crossings(T2s, np.array(E3["bit3"]), np.array(E3["phase3"]))
plt.figure(figsize=(5.2, 3.6))
for c, st in zip(c5, ["k--", "C0o-", "C1s-", "C2^-", "C3v-"]): plt.plot(T2s, E3[c], st, label=c, ms=4)
plt.xlabel(r"$T_2$ ($\mu$s), $T_1$=80 $\mu$s, 3 $\mu$s storage"); plt.ylabel("basis-averaged logical error"); plt.legend(fontsize=7)
plt.title("Thermal noise on every gate"); save("fig3_thermal"); log(f"E3 done, bit/phase crossover T2 = {R['bit_phase_crossover_T2']}")

# ---- E4 memory window -------------------------------------------------------------------
tt = np.array([1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, 500, 800, 1200, 2000, 3000], float)
if a.quick: tt = tt[::2]
cohs = [(500, 500), (1000, 1000), (5000, 5000)]
R["memory"] = {}
fig, axs = plt.subplots(1, 3, figsize=(10, 3.3), sharey=True)
for ax, (T1, T2) in zip(axs, cohs):
    cv = {c: memory.curve(c, T1, T2, tt) for c in ["none", "phase3", "shor9"]}
    win = memory.crossings(tt, cv["shor9"], cv["none"])
    R["memory"][f"T1=T2={T1}us"] = {"t_us": tt.tolist(), **{c: v.tolist() for c, v in cv.items()},
        "shor_beats_unencoded_between_us": win,
        "t_max_us": {c: {f"Qchan={qc}": memory.t_max(tt, cv[c], qc) for qc in (0.0, 0.03)} for c in cv}}
    for c, st in zip(cv, ["k--", "C1s-", "C3d-"]): ax.loglog(tt, cv[c], st, label=c, ms=3)
    ax.axhline(K.Q_TH, ls=":", c="gray"); ax.set_title(f"T1=T2={T1} us", fontsize=9); ax.set_xlabel("storage time (us)")
axs[0].set_ylabel("memory QBER (basis-averaged)"); axs[0].legend(fontsize=7); save("fig4_memory"); log("E4 done")

# ---- E5 decoy / GLLP / naive / finite key ----------------------------------------------
Ls = np.linspace(0, 200, 401); mu, nu = 0.5, 0.1
naive = [K.rate_naive(mu, L) for L in Ls]; dec = [K.rate_decoy(mu, nu, L) for L in Ls]
gl = [K.best_mu_gllp(L)[1] for L in Ls]
R["distance_km"] = {"naive_insecure": K.max_distance(lambda L: K.rate_naive(mu, L), 0, 300),
                    "decoy": K.max_distance(lambda L: K.rate_decoy(mu, nu, L), 0, 300),
                    "gllp_no_decoy_optimal_mu": K.max_distance(lambda L: K.best_mu_gllp(L)[1], 0, 200)}
plt.figure(figsize=(5.2, 3.6))
for y, lab, st in [(naive, "naive (insecure)", "C3-"), (dec, "decoy-state", "C0-"), (gl, "GLLP, no decoy (best mu)", "C2-")]:
    plt.semilogy(Ls, np.maximum(y, 1e-12), st, label=lab)
plt.ylim(1e-9, 1e-1); plt.xlabel("fibre length (km)"); plt.ylabel("secret bits / pulse"); plt.legend(fontsize=7)
plt.title("What decoy states actually buy"); save("fig5_decoy")
qs = [0.01, 0.02, 0.03, 0.05, 0.07, 0.09]
R["finite_key"] = {"qber": qs, "min_block_N": [K.min_block(x) for x in qs], "asymptotic_rate": [float(K.sp_rate(x)) for x in qs]}
Ns = np.logspace(3, 7, 40); plt.figure(figsize=(5.2, 3.6))
for qv in (0.01, 0.03, 0.05, 0.07): plt.semilogx(Ns, [K.finite_key_fraction(int(n), qv) for n in Ns], label=f"QBER {qv:.0%}")
plt.xlabel("sifted block length N"); plt.ylabel("secret bits per sifted bit"); plt.legend(fontsize=7); plt.title("Finite-size penalty (illustrative)")
save("fig6_finitekey"); log("E5 done")

# ---- E6 Steane purification (exact) -----------------------------------------------------
pp = np.linspace(0, 0.3, 61)
R["steane"] = {"p": pp.tolist(), "purified": [steane.qber_purified(p) for p in pp], "raw": [steane.qber_raw(p) for p in pp],
               "pseudo_threshold_p": steane.pseudo_threshold()}
R["steane"]["pseudo_threshold_raw_qber"] = 2 * R["steane"]["pseudo_threshold_p"] / 3
plt.figure(figsize=(5.2, 3.6)); plt.plot(pp, R["steane"]["raw"], "C0-", label="raw pair (2p/3)"); plt.plot(pp, R["steane"]["purified"], "C3-", label="Steane-purified (exact)")
plt.axhline(K.Q_TH, ls=":", c="gray"); plt.xlabel("depolarizing p"); plt.ylabel("QBER"); plt.legend(fontsize=7); plt.title("Entanglement purification")
save("fig7_steane"); log("E6 done")
R["Q_TH"] = K.Q_TH
json.dump(R, open("results/results.json", "w"), indent=1, default=float)
log("ALL DONE -> results/results.json, figures/")
