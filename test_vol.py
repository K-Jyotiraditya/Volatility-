"""Unit tests for the implied-vol solvers and SABR calibration."""
from __future__ import annotations

import numpy as np
import pytest

from arbitrage import (butterfly_report, calendar_report,
                       calibrate_sabr_arbfree, risk_neutral_density,
                       black76_calls)
from black_scholes import bs_price, bs_vega, price_bounds
from implied_vol import implied_vol, implied_vol_brent, implied_vol_newton
from sabr import hagan_lognormal_vol, calibrate_sabr

S, r, q, T = 100.0, 0.03, 0.0, 1.0


# --------------------------------------------------------------------------- #
# Implied-vol inversion
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("K", [70, 85, 100, 115, 130])
@pytest.mark.parametrize("sigma", [0.10, 0.25, 0.60])
def test_inversion_round_trip(K, sigma):
    for opt in ("call", "put"):
        price = bs_price(S, K, r, sigma, T, opt, q)
        assert abs(implied_vol(price, S, K, r, T, opt, q) - sigma) < 1e-6


def test_newton_and_brent_agree():
    for K in (60, 90, 100, 110, 140):
        price = bs_price(S, K, r, 0.3, T, "call", q)
        iv_n = implied_vol_newton(price, S, K, r, T, "call", q)
        iv_b = implied_vol_brent(price, S, K, r, T, "call", q)
        assert abs(iv_n - iv_b) < 1e-6


def test_deep_otm_does_not_blow_up():
    # A 60% OTM call far in the wing: vega is minuscule, the regime that breaks
    # a naive Newton step. The safeguarded solver must stay finite and accurate.
    K = 160.0
    price = bs_price(S, K, r, 0.45, T, "call", q)
    iv = implied_vol(price, S, K, r, T, "call", q)
    assert np.isfinite(iv) and abs(iv - 0.45) < 1e-4


def test_price_outside_bounds_returns_nan():
    lo, hi = price_bounds(S, 100, r, T, "call", q)
    assert np.isnan(implied_vol(hi * 1.01, S, 100, r, T, "call", q))
    assert np.isnan(implied_vol(max(lo - 1.0, -1.0), S, 100, r, T, "call", q))


def test_vega_peaks_near_the_money():
    near = bs_vega(S, 100, r, 0.2, T, q)
    far = bs_vega(S, 160, r, 0.2, T, q)
    assert near > far


# --------------------------------------------------------------------------- #
# SABR
# --------------------------------------------------------------------------- #
def test_hagan_atm_limit_is_finite():
    # K = F triggers the z/x(z) → 1 removable singularity; must not be NaN.
    F = 100.0
    v = hagan_lognormal_vol(F, np.array([F]), T, 2.0, 0.5, -0.3, 0.4)
    assert np.isfinite(v).all() and v[0] > 0


def test_sabr_calibration_recovers_parameters():
    F, beta = 100.0, 0.5
    true = dict(alpha=2.0, beta=beta, rho=-0.3, nu=0.45)
    strikes = np.linspace(70, 135, 21)
    vols = hagan_lognormal_vol(F, strikes, T, true["alpha"], beta,
                               true["rho"], true["nu"])
    fit = calibrate_sabr(F, T, strikes, vols, beta=beta)
    assert fit["rmse"] < 1e-5
    assert abs(fit["alpha"] - true["alpha"]) < 1e-2
    assert abs(fit["rho"] - true["rho"]) < 1e-2
    assert abs(fit["nu"] - true["nu"]) < 1e-2


def test_sabr_produces_a_skew():
    # Negative rho must tilt the smile: low strikes richer than high strikes.
    F = 100.0
    lo = hagan_lognormal_vol(F, np.array([80.0]), T, 2.0, 0.5, -0.4, 0.5)[0]
    hi = hagan_lognormal_vol(F, np.array([120.0]), T, 2.0, 0.5, -0.4, 0.5)[0]
    assert lo > hi


# --------------------------------------------------------------------------- #
# Static no-arbitrage diagnostics
# --------------------------------------------------------------------------- #
def test_healthy_smile_is_butterfly_free():
    F, beta = 100.0, 0.8
    p = {"alpha": 0.2 * F ** (1 - beta), "rho": -0.3, "nu": 0.4}
    rep = butterfly_report(F, 1.0, p, beta, r=0.03)
    assert rep["arbitrage_free"]
    assert rep["min_density"] > -1e-6


def test_extreme_wings_trigger_butterfly_arbitrage():
    # Short maturity + very high vol-of-vol is where Hagan's expansion bends the
    # wings into a negative density — the detector must flag it.
    F, beta = 100.0, 0.5
    p = {"alpha": 0.25 * F ** (1 - beta), "rho": -0.7, "nu": 6.0}
    rep = butterfly_report(F, 0.25, p, beta, r=0.03, k_lo=0.4, k_hi=2.2, n=400)
    assert not rep["arbitrage_free"]
    assert rep["min_density"] < 0


def test_risk_neutral_density_integrates_to_one():
    F, T, beta = 100.0, 1.0, 0.8
    vols = hagan_lognormal_vol(F, np.linspace(20, 320, 1500), T,
                               0.2 * F ** (1 - beta), beta, -0.3, 0.4)
    strikes = np.linspace(20, 320, 1500)
    calls = black76_calls(F, T, strikes, vols, r=0.03)
    g = risk_neutral_density(strikes, calls, 0.03, T)
    assert abs(np.trapezoid(g, strikes) - 1.0) < 1e-2


def test_increasing_term_structure_is_calendar_free():
    beta, r = 0.8, 0.03
    tenors = [0.1, 0.5, 1.0, 2.0]
    forwards = [100 * np.exp(r * t) for t in tenors]
    pl = [{"alpha": 0.2 * f ** (1 - beta), "rho": -0.3, "nu": 0.4} for f in forwards]
    assert calendar_report(forwards, tenors, pl, beta)["arbitrage_free"]


def test_arbfree_calibration_matches_clean_smile():
    F, T, beta = 100.0, 1.0, 0.5
    true = {"alpha": 2.0, "rho": -0.3, "nu": 0.45}
    strikes = np.linspace(70, 135, 21)
    vols = hagan_lognormal_vol(F, strikes, T, true["alpha"], beta,
                               true["rho"], true["nu"])
    fit = calibrate_sabr_arbfree(F, T, strikes, vols, beta=beta, penalty=5.0)
    # A clean smile is already arbitrage-free, so the penalty is inactive and the
    # fit must still recover the parameters tightly.
    assert fit["rmse"] < 1e-3
    assert abs(fit["nu"] - true["nu"]) < 0.05


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
