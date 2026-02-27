"""Market-data sourcing: a reproducible synthetic generator and a live
``yfinance`` loader.

The synthetic generator is the default. It builds each tenor's smile from a
*known* SABR parameter set, converts those vols to option prices, and adds a
touch of quote noise. That gives the rest of the pipeline a ground truth: the
solver must recover the vols and the calibrator must recover (approximately)
the SABR parameters that produced them — a closed validation loop you cannot
get from live quotes.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from black_scholes import bs_price
from sabr import hagan_lognormal_vol


def synthetic_chain(S0=100.0, r=0.03, q=0.0,
                    tenors=(0.08, 0.25, 0.5, 1.0, 2.0),
                    n_strikes=17, width=0.45, noise=0.002, seed=0):
    """Generate an option chain from a term structure of SABR smiles.

    Returns a tidy DataFrame: ``[T, forward, strike, type, price, iv_true]``.
    Out-of-the-money options are quoted (calls above the forward, puts below) —
    the liquid, well-conditioned side of the market.
    """
    rng = np.random.default_rng(seed)
    beta = 0.8  # equity-like backbone
    rows = []
    for T in tenors:
        F = S0 * np.exp((r - q) * T)
        # A plausible term structure: vol drifts down, skew and curvature decay.
        atm = 0.22 - 0.02 * T
        alpha = atm * F ** (1 - beta)
        rho = -0.45 + 0.05 * T
        nu = 0.35 + 0.30 * np.exp(-T)

        strikes = np.linspace(F * (1 - width), F * (1 + width), n_strikes)
        iv = hagan_lognormal_vol(F, strikes, T, alpha, beta, rho, nu)
        iv_quoted = iv + rng.normal(0.0, noise, size=iv.shape)

        for K, v_true, v_q in zip(strikes, iv, iv_quoted):
            opt = "call" if K >= F else "put"
            price = bs_price(S0, K, r, v_q, T, opt, q)
            rows.append((T, F, K, opt, price, v_true))

    return pd.DataFrame(rows, columns=["T", "forward", "strike", "type",
                                       "price", "iv_true"])


def load_yfinance_chain(ticker, r=0.03, q=0.0, max_expiries=6, today=None):
    """Pull a live option chain via ``yfinance`` into the same schema.

    Mid prices are used and obviously stale/illiquid quotes (zero bid, crossed
    markets) are dropped. Raises a clear error on any network/parse failure so
    the caller can fall back to the synthetic generator.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("yfinance is not installed.") from exc

    today = today or dt.date.today()
    tk = yf.Ticker(ticker)

    try:
        spot = float(tk.history(period="1d")["Close"].iloc[-1])
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(f"Could not fetch spot for {ticker}: {exc}") from exc

    expiries = list(tk.options)[:max_expiries]
    if not expiries:
        raise RuntimeError(f"No listed expiries for {ticker}.")

    rows = []
    for e in expiries:
        T = (dt.date.fromisoformat(e) - today).days / 365.0
        if T <= 0:
            continue
        F = spot * np.exp((r - q) * T)
        chain = tk.option_chain(e)
        for frame, opt in ((chain.calls, "call"), (chain.puts, "put")):
            bid, ask = frame["bid"].to_numpy(), frame["ask"].to_numpy()
            strike = frame["strike"].to_numpy()
            mid = 0.5 * (bid + ask)
            keep = (bid > 0) & (ask > 0) & (ask >= bid)
            # Only the OTM wing of each option type carries clean information.
            keep &= (strike >= F) if opt == "call" else (strike < F)
            for K, p in zip(strike[keep], mid[keep]):
                rows.append((T, F, float(K), opt, float(p), np.nan))

    if not rows:
        raise RuntimeError(f"No usable quotes returned for {ticker}.")

    df = pd.DataFrame(rows, columns=["T", "forward", "strike", "type",
                                     "price", "iv_true"])
    df.attrs["spot"] = spot
    return df
