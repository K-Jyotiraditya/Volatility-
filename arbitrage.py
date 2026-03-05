"""Static no-arbitrage diagnostics for a fitted volatility surface.

A surface can be smooth and still admit free money. Two static conditions must
hold (Gatheral, *The Volatility Surface*):

* **No butterfly arbitrage** — the implied risk-neutral density must be
  non-negative. By Breeden-Litzenberger the density is
  ``g(K) = e^{rT} ∂²C/∂K²``, so this is exactly convexity of the call price in
  strike. A dip below zero means a butterfly spread (long 2 wings, short 2
  bodies) has negative cost and non-negative payoff: arbitrage.

* **No calendar arbitrage** — total implied variance ``w(k, T) = σ²(k, T)·T``
  must be non-decreasing in maturity at fixed forward-log-moneyness ``k =
  ln(K/F)``. If a longer-dated option had less total variance than a shorter one
  at the same moneyness, a calendar spread would lock in a riskless profit.

This module reports both, and offers a calibration that *penalises* butterfly
violations so SABR cannot fit itself into an arbitrageable corner of the wings.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import norm

from sabr import hagan_lognormal_vol


def black76_calls(F, T, strikes, vols, r):
    """Undiscounted-forward Black-76 call prices for a vol smile.

    SABR is a forward-measure model, so Black-76 (rather than spot Black-Scholes)
    is the natural pricer; the discount factor is constant in K and so does not
    affect the convexity test.
    """
    vt = vols * np.sqrt(T)
    d1 = (np.log(F / strikes) + 0.5 * vt ** 2) / vt
    d2 = d1 - vt
    return np.exp(-r * T) * (F * norm.cdf(d1) - strikes * norm.cdf(d2))


def sabr_call_prices(F, T, strikes, params, beta, r):
    vols = hagan_lognormal_vol(F, strikes, T, params["alpha"], beta,
                               params["rho"], params["nu"])
    return black76_calls(F, T, strikes, vols, r)


def risk_neutral_density(strikes, call_prices, r, T):
    """Breeden-Litzenberger density ``g(K) = e^{rT} ∂²C/∂K²`` (finite difference)."""
    d2 = np.gradient(np.gradient(call_prices, strikes), strikes)
    return np.exp(r * T) * d2


def butterfly_report(F, T, params, beta, r, k_lo=0.55, k_hi=1.75, n=240):
    """Check convexity of the SABR call curve. Returns the density and a verdict.

    Edge points are trimmed: the twice-differenced density is unreliable at the
    very boundary of the strike grid.
    """
    strikes = np.linspace(k_lo * F, k_hi * F, n)
    calls = sabr_call_prices(F, T, strikes, params, beta, r)
    density = risk_neutral_density(strikes, calls, r, T)
    core = density[3:-3]
    return {"strikes": strikes, "density": density,
            "min_density": float(core.min()),
            "arbitrage_free": bool(core.min() > -1e-6),
            "integral": float(np.trapezoid(np.maximum(density, 0.0), strikes))}


def calendar_report(forwards, tenors, params_list, beta, k_grid=None):
    """Check that total variance is non-decreasing in maturity at fixed moneyness."""
    if k_grid is None:
        k_grid = np.linspace(-0.30, 0.30, 25)
    order = np.argsort(tenors)
    tenors = np.asarray(tenors)[order]
    forwards = np.asarray(forwards)[order]
    params_list = [params_list[i] for i in order]

    w = np.empty((len(tenors), k_grid.size))
    for i, (F, T, p) in enumerate(zip(forwards, tenors, params_list)):
        K = F * np.exp(k_grid)
        vol = hagan_lognormal_vol(F, K, T, p["alpha"], beta, p["rho"], p["nu"])
        w[i] = vol ** 2 * T
    dw = np.diff(w, axis=0)               # change in total variance across tenor
    return {"total_variance": w, "k_grid": k_grid, "tenors": tenors,
            "min_increment": float(dw.min()),
            "arbitrage_free": bool(dw.min() > -1e-8)}


def calibrate_sabr_arbfree(F, T, strikes, market_vols, beta=0.8, penalty=5.0,
                           weights=None):
    """Calibrate SABR with a soft penalty on butterfly (density) violations.

    The residual vector is augmented with ``penalty · max(−g, 0)`` sampled on a
    strike grid, so the optimiser is pushed away from parameter sets whose wings
    imply a negative density — without hard constraints that could make the fit
    brittle.
    """
    strikes = np.asarray(strikes, float)
    market_vols = np.asarray(market_vols, float)
    w = np.ones_like(market_vols) if weights is None else np.asarray(weights)
    atm = float(np.interp(F, strikes, market_vols))
    grid = np.linspace(0.6 * F, 1.7 * F, 120)

    def residual(p):
        a, rho, nu = p
        params = {"alpha": a, "rho": rho, "nu": nu}
        fit_err = w * (hagan_lognormal_vol(F, strikes, T, a, beta, rho, nu) - market_vols)
        if penalty <= 0:
            return fit_err
        calls = sabr_call_prices(F, T, grid, params, beta, r=0.0)
        density = risk_neutral_density(grid, calls, 0.0, T)
        arb = penalty * np.maximum(-density[3:-3], 0.0)
        return np.concatenate([fit_err, arb])

    x0 = [atm * F ** (1 - beta), -0.3, 0.4]
    bounds = ([1e-6, -0.999, 1e-6], [5.0, 0.999, 10.0])
    sol = least_squares(residual, x0, bounds=bounds, method="trf", xtol=1e-12)
    a, rho, nu = sol.x
    vol_resid = hagan_lognormal_vol(F, strikes, T, a, beta, rho, nu) - market_vols
    return {"alpha": float(a), "beta": float(beta), "rho": float(rho),
            "nu": float(nu), "rmse": float(np.sqrt(np.mean(vol_resid ** 2)))}
