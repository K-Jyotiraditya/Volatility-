"""SABR stochastic-volatility smile model (Hagan et al., 2002).

SABR posits a forward and its volatility driven by correlated Brownian motions,

    dF = α F^β dW,   dα = ν α dZ,   d⟨W, Z⟩ = ρ dt,

and Hagan's singular-perturbation expansion gives a closed-form *Black implied
volatility* as a function of strike. The four parameters carry clean intuition:

* α  — overall level of volatility (anchors the ATM vol),
* β  — backbone elasticity (β=1 lognormal, β=0 normal); usually fixed a priori
       because it is only weakly identified jointly with ρ,
* ρ  — spot/vol correlation, which tilts the smile into a skew,
* ν  — vol-of-vol, which controls the smile's curvature.

We calibrate (α, ρ, ν) per tenor to a market smile by nonlinear least squares.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def hagan_lognormal_vol(F, K, T, alpha, beta, rho, nu):
    """Hagan's lognormal (Black) implied vol. Vectorised over ``K``.

    The removable singularities at the money (z→0) and at β→1 are handled by
    falling back to the analytic limits rather than dividing by zero.
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    one_minus_b = 1.0 - beta

    log_fk = np.log(F / K)
    fk_pow = (F * K) ** (one_minus_b / 2.0)

    # Denominator series in ln(F/K): equals fk_pow at the money.
    denom = fk_pow * (1.0
                      + one_minus_b ** 2 / 24.0 * log_fk ** 2
                      + one_minus_b ** 4 / 1920.0 * log_fk ** 4)

    z = (nu / alpha) * fk_pow * log_fk
    sqrt_term = np.sqrt(1.0 - 2.0 * rho * z + z ** 2)
    x_z = np.log((sqrt_term + z - rho) / (1.0 - rho))
    # z/x(z) → 1 as z → 0; compute the ratio under errstate because the ATM
    # entries are a genuine 0/0 that we immediately overwrite with the limit.
    with np.errstate(divide="ignore", invalid="ignore"):
        z_over_x = np.where(np.abs(z) < 1e-10, 1.0, z / x_z)

    # Time-to-maturity correction (the curly-brace term in Hagan's paper).
    correction = 1.0 + (
        one_minus_b ** 2 / 24.0 * alpha ** 2 / (F * K) ** one_minus_b
        + 0.25 * rho * beta * nu * alpha / fk_pow
        + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2
    ) * T

    return alpha / denom * z_over_x * correction


def calibrate_sabr(F, T, strikes, market_vols, beta=0.5, fit_beta=False,
                   weights=None):
    """Fit SABR to one tenor's smile by nonlinear least squares.

    Returns a dict with the calibrated parameters and the fit RMSE (in vol
    points). ``beta`` is fixed by default — the industry-standard choice, since
    β and ρ trade off against each other and β is better pinned by the asset
    class (≈0.5 for rates, ≈1 for FX/equity) than by the data.
    """
    strikes = np.asarray(strikes, dtype=float)
    market_vols = np.asarray(market_vols, dtype=float)
    w = np.ones_like(market_vols) if weights is None else np.asarray(weights)

    # ATM vol seeds α via the leading-order relation σ_ATM ≈ α / F^{1-β}.
    atm_vol = float(np.interp(F, strikes, market_vols))

    if fit_beta:
        x0 = [atm_vol * F ** (1 - 0.5), 0.5, -0.3, 0.4]
        bounds = ([1e-6, 0.0, -0.999, 1e-6], [5.0, 1.0, 0.999, 10.0])

        def residual(p):
            a, b, rho, nu = p
            return w * (hagan_lognormal_vol(F, strikes, T, a, b, rho, nu) - market_vols)
    else:
        x0 = [atm_vol * F ** (1 - beta), -0.3, 0.4]
        bounds = ([1e-6, -0.999, 1e-6], [5.0, 0.999, 10.0])

        def residual(p):
            a, rho, nu = p
            return w * (hagan_lognormal_vol(F, strikes, T, a, beta, rho, nu) - market_vols)

    sol = least_squares(residual, x0, bounds=bounds, method="trf", xtol=1e-12)

    if fit_beta:
        alpha, beta, rho, nu = sol.x
    else:
        alpha, rho, nu = sol.x

    rmse = float(np.sqrt(np.mean(sol.fun ** 2)))
    return {"alpha": float(alpha), "beta": float(beta), "rho": float(rho),
            "nu": float(nu), "rmse": rmse}
