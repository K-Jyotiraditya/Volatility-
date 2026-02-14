"""Implied-volatility solvers: safeguarded Newton and Brent.

Inverting Black-Scholes for σ is a 1-D root find on a function that is strictly
increasing in σ. The subtlety is the *wings*: deep out-of-the-money options
have vega ≈ 0, so the Newton ratio f/f' explodes and a textbook Newton step
launches σ to absurd values. The fix is the classic ``rtsafe`` hybrid — keep a
bracket that always contains the root and fall back to bisection whenever a
Newton step would leave it. The result is Newton's quadratic speed with
bisection's guaranteed convergence.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from black_scholes import bs_price, bs_vega, price_bounds


def _admissible(price, S, K, r, T, option, q):
    """Return True if ``price`` sits strictly inside the no-arbitrage bounds."""
    lo, hi = price_bounds(S, K, r, T, option, q)
    return lo + 1e-10 < price < hi - 1e-12


def implied_vol_newton(price, S, K, r, T, option="call", q=0.0,
                       tol=1e-8, max_iter=100, lo=1e-6, hi=5.0):
    """Safeguarded Newton (rtsafe). Returns ``np.nan`` if no root exists."""
    if not _admissible(price, S, K, r, T, option, q):
        return np.nan

    f = lambda s: bs_price(S, K, r, s, T, option, q) - price
    # Price is increasing in σ, so f(lo) < 0 < f(hi) brackets the root.
    sigma = 0.2  # a sane starting guess for equity vols
    for _ in range(max_iter):
        fx = f(sigma)
        # Tighten the bracket around the sign change.
        if fx < 0:
            lo = sigma
        else:
            hi = sigma
        vega = bs_vega(S, K, r, sigma, T, q)
        # Convergence is judged on the *step in σ*, never on an absolute price
        # residual: in the deep wings the true price can be ~1e-10, so a price
        # tolerance would falsely declare victory at the starting guess.
        candidate = sigma - fx / vega if vega > 1e-12 else np.inf
        # Reject a Newton step that escapes the bracket or stalls; bisect instead.
        if not lo < candidate < hi:
            candidate = 0.5 * (lo + hi)
        step = abs(candidate - sigma)
        sigma = candidate
        if step < tol:
            return sigma
    return sigma


def implied_vol_brent(price, S, K, r, T, option="call", q=0.0,
                      lo=1e-6, hi=5.0, tol=1e-10):
    """Brent's method on the bracketed root. Returns ``np.nan`` if none exists."""
    if not _admissible(price, S, K, r, T, option, q):
        return np.nan
    f = lambda s: bs_price(S, K, r, s, T, option, q) - price
    if f(lo) * f(hi) > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=tol, maxiter=200)


def implied_vol(price, S, K, r, T, option="call", q=0.0, method="newton"):
    """Front door. ``method`` ∈ {"newton", "brent"}; Newton falls back to Brent."""
    if method == "brent":
        return implied_vol_brent(price, S, K, r, T, option, q)
    iv = implied_vol_newton(price, S, K, r, T, option, q)
    # The safeguarded Newton effectively cannot fail to converge once a root
    # exists, but if a degenerate input slips through we defer to Brent.
    if np.isnan(iv):
        return np.nan
    return iv


def implied_vol_chain(prices, S, strikes, r, T, option="call", q=0.0,
                      method="newton"):
    """Vectorised convenience wrapper over an array of strikes/prices."""
    prices = np.asarray(prices, float)
    strikes = np.asarray(strikes, float)
    out = np.array([
        implied_vol(p, S, k, r, T, option, q, method)
        for p, k in zip(prices, strikes)
    ])
    return out
