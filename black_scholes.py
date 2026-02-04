"""Black-Scholes-Merton pricing and risk, self-contained for this project.

Everything the volatility tooling needs — price, vega, and the analytic
no-arbitrage price bounds — lives here so the surface builder never reaches
outside its own directory.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import norm


def _d1_d2(S, K, r, sigma, T, q=0.0):
    vol_t = sigma * np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / vol_t
    return d1, d1 - vol_t


def bs_price(S, K, r, sigma, T, option="call", q=0.0):
    """Black-Scholes-Merton price with continuous dividend yield ``q``."""
    d1, d2 = _d1_d2(S, K, r, sigma, T, q)
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    if option == "call":
        return S * df_q * norm.cdf(d1) - K * df_r * norm.cdf(d2)
    return K * df_r * norm.cdf(-d2) - S * df_q * norm.cdf(-d1)


def bs_vega(S, K, r, sigma, T, q=0.0):
    """Sensitivity of price to volatility — the Newton step's denominator.

    Vega collapses to zero for deep in/out-of-the-money strikes, which is
    precisely why a naive Newton iteration blows up there and a safeguarded
    bracket is required.
    """
    d1, _ = _d1_d2(S, K, r, sigma, T, q)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)


def price_bounds(S, K, r, T, option="call", q=0.0):
    """No-arbitrage (lower, upper) price interval for a European option.

    Outside this interval no real implied volatility exists, so the solvers use
    it to bail out gracefully instead of thrashing.
    """
    df_q, df_r = np.exp(-q * T), np.exp(-r * T)
    if option == "call":
        return max(S * df_q - K * df_r, 0.0), S * df_q
    return max(K * df_r - S * df_q, 0.0), K * df_r
