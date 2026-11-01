"""Key-rate analytics: Shor-Preskill asymptotics, decoy-state (Ma-Qi-Zhao-Lo 2005), GLLP, finite size."""
import numpy as np
from scipy.optimize import brentq, minimize_scalar

def h2(x):
    x = np.clip(x, 1e-12, 1 - 1e-12)
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)

Q_TH = brentq(lambda q: 1 - 2 * h2(q), 1e-4, 0.3)       # Shor-Preskill one-way threshold (~0.110)

def sp_rate(q):
    """Asymptotic BB84 rate per sifted bit, ideal single photons, one-way post-processing."""
    return 1 - 2 * h2(q)

# ---------------------------------------------------------------- fibre / detector model
P = dict(alpha=0.21, eta_det=0.15, Y0=1.7e-6, e_det=0.033, e0=0.5, f=1.16, q=0.5)

def eta(L, par=P):
    return par["eta_det"] * 10 ** (-par["alpha"] * L / 10)

def gains(mu, L, par=P):
    """Overall gain Q_mu and QBER E_mu for a Poisson (weak coherent) source."""
    t = eta(L, par); Y0 = par["Y0"]
    Q = 1 - (1 - Y0) * np.exp(-t * mu)
    EQ = par["e0"] * Y0 + par["e_det"] * (1 - np.exp(-t * mu))
    return Q, EQ / Q

def rate_naive(mu, L, par=P):
    """INSECURE reference: treats every click as if it were single-photon."""
    Q, E = gains(mu, L, par)
    return par["q"] * Q * (1 - h2(E) - par["f"] * h2(E))

def rate_gllp(mu, L, par=P):
    """Secure WITHOUT decoy states (GLLP): worst case, Eve controls all multi-photon pulses."""
    Q, E = gains(mu, L, par)
    p_multi = 1 - np.exp(-mu) * (1 + mu)
    delta = p_multi / Q                                  # tagged fraction of clicks (upper bound)
    om = 1 - delta
    if om <= 0 or E / om >= 0.5: return 0.0
    return max(0.0, par["q"] * (-Q * par["f"] * h2(E) + Q * om * (1 - h2(E / om))))

def rate_decoy(mu, nu, L, par=P):
    """Vacuum + weak decoy, asymptotic infinite decoy statistics (Ma et al. 2005)."""
    Q, E = gains(mu, L, par); Qn, En = gains(nu, L, par); Y0 = par["Y0"]
    y1 = mu / (mu * nu - nu**2) * (Qn * np.exp(nu) - Q * np.exp(mu) * nu**2 / mu**2 - (mu**2 - nu**2) / mu**2 * Y0)
    if y1 <= 0: return 0.0
    e1 = (En * Qn * np.exp(nu) - par["e0"] * Y0) / (y1 * nu)
    if e1 >= 0.5: return 0.0
    Q1 = mu * np.exp(-mu) * y1
    return max(0.0, par["q"] * (-Q * par["f"] * h2(E) + Q1 * (1 - h2(e1))))

def max_distance(fn, lo=0.0, hi=400.0):
    Ls = np.linspace(lo, hi, 1601); r = np.array([fn(L) for L in Ls])
    ok = np.where(r > 0)[0]
    return float(Ls[ok[-1]]) if len(ok) else 0.0

def best_mu_gllp(L, par=P):
    mus = np.logspace(-5, 0, 400)
    r = np.array([rate_gllp(m, L, par) for m in mus]); i = int(r.argmax())
    return float(mus[i]), float(r[i])

# ---------------------------------------------------------------- finite-size (illustrative)
def finite_key_fraction(n, qber, eps_sec=1e-10, eps_cor=1e-15, f=1.16, k_frac=0.1):
    """Secret bits per sifted bit for ideal single-photon BB84 with a finite block of N sifted bits.
    ILLUSTRATIVE simplified model after Tomamichel-Lim-Gisin-Renner 2012 / Lim et al. 2014:
      n = (1-k_frac)*N key bits, k = k_frac*N test bits; statistical shift mu on the QBER;
      l = n[1-h(Q+mu)] - leak_EC - 6log2(21/eps_sec) - log2(2/eps_cor). Not a certified bound."""
    N = n; k = max(int(k_frac * N), 1); nk = N - k
    mu = np.sqrt((nk + k) / (nk * k) * (k + 1) / k * np.log(1 / eps_sec))
    leak = f * nk * h2(qber)
    l = nk * (1 - h2(qber + mu)) - leak - 6 * np.log2(21 / eps_sec) - np.log2(2 / eps_cor)
    return max(0.0, l / N)

def min_block(qber, **kw):
    for e in np.arange(2, 9, 0.02):
        N = int(10 ** e)
        if finite_key_fraction(N, qber, **kw) > 0: return N
    return None

def multi_share(mu, L, par=P, nmax=40):
    """Fraction of detection events caused by pulses with >= 2 photons."""
    from math import factorial
    t = eta(L, par); Y0 = par["Y0"]
    Y = lambda n: 1 - (1 - Y0) * (1 - t) ** n
    Q = sum(np.exp(-mu) * mu**n / factorial(n) * Y(n) for n in range(nmax))
    return sum(np.exp(-mu) * mu**n / factorial(n) * Y(n) for n in range(2, nmax)) / Q
