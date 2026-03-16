"""End-to-end volatility-surface construction.

    market quotes  ->  implied vols  ->  per-tenor SABR fit  ->  3D surface

By default the script runs on a reproducible synthetic market (so the solver
and calibrator can be validated against ground truth). Pass a ticker to pull a
live chain instead::

    python main.py            # synthetic market
    python main.py SPY        # live chain via yfinance (needs network)

Outputs ``sabr_smiles.png`` (per-tenor fits) and ``vol_surface.png`` (3D).
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from arbitrage import butterfly_report, calendar_report
from implied_vol import implied_vol
from market_data import synthetic_chain, load_yfinance_chain
from sabr import calibrate_sabr, hagan_lognormal_vol


def build_market(argv):
    """Return ``(dataframe, spot, r, q, label)`` from CLI args."""
    r, q = 0.03, 0.0
    if len(argv) > 1:
        ticker = argv[1].upper()
        try:
            df = load_yfinance_chain(ticker, r, q)
            return df, df.attrs["spot"], r, q, f"{ticker} (live)"
        except Exception as exc:
            print(f"[warn] live fetch failed ({exc}); using synthetic market.")
    return synthetic_chain(r=r, q=q), 100.0, r, q, "synthetic"


def main():
    df, spot, r, q, label = build_market(sys.argv)
    print(f"Market: {label}  |  spot={spot:.2f}  |  {len(df)} quotes  "
          f"|  {df['T'].nunique()} tenors")

    # --- Invert every quote to a Black-Scholes implied vol ----------------- #
    df = df.copy()
    df["iv"] = [
        implied_vol(row.price, spot, row.strike, r, row.T, row.type, q)
        for row in df.itertuples()
    ]
    df = df.dropna(subset=["iv"])
    if df["iv_true"].notna().any():
        err = (df["iv"] - df["iv_true"]).abs().max()
        print(f"Solver vs ground-truth vols: max abs error = {err:.2e}")

    # --- Calibrate SABR per tenor ----------------------------------------- #
    print("\nPer-tenor SABR calibration (beta fixed at 0.8):")
    print(f"{'T':>6} {'alpha':>8} {'beta':>6} {'rho':>8} {'nu':>8} {'rmse':>9}")
    tenors = np.sort(df["T"].unique())
    params = {}
    for T in tenors:
        sub = df[df["T"] == T].sort_values("strike")
        F = sub["forward"].iloc[0]
        fit = calibrate_sabr(F, T, sub["strike"].to_numpy(),
                             sub["iv"].to_numpy(), beta=0.8)
        params[T] = fit
        print(f"{T:6.2f} {fit['alpha']:8.4f} {fit['beta']:6.2f} "
              f"{fit['rho']:8.4f} {fit['nu']:8.4f} {fit['rmse']:9.2e}")

    # --- Static no-arbitrage audit ---------------------------------------- #
    print("\nStatic no-arbitrage audit:")
    forwards = [df[df["T"] == T]["forward"].iloc[0] for T in tenors]
    bf = {T: butterfly_report(forwards[i], T, params[T], 0.8, r)
          for i, T in enumerate(tenors)}
    cal = calendar_report(forwards, tenors, [params[T] for T in tenors], 0.8)
    for T in tenors:
        flag = "OK " if bf[T]["arbitrage_free"] else "ARB"
        print(f"  T={T:4.2f}  butterfly {flag}  min density {bf[T]['min_density']:+.5f}")
    print(f"  calendar {'OK ' if cal['arbitrage_free'] else 'ARB'}  "
          f"min total-variance increment {cal['min_increment']:+.5f}")

    fig_a, (axd, axc) = plt.subplots(1, 2, figsize=(12, 4.6))
    for T in tenors:
        rep = bf[T]
        axd.plot(rep["strikes"] / np.interp(T, tenors, forwards),
                 rep["density"], label=f"T={T:g}y")
    axd.axhline(0, color="k", lw=0.8)
    axd.set(title="Risk-neutral density (butterfly check)",
            xlabel="moneyness K/F", ylabel=r"$e^{rT}\,\partial^2C/\partial K^2$")
    axd.legend(fontsize=8)
    for j, k in enumerate(cal["k_grid"][::6]):
        axc.plot(cal["tenors"], cal["total_variance"][:, j * 6], marker="o",
                 label=f"k={k:+.2f}")
    axc.set(title="Total variance vs maturity (calendar check)",
            xlabel="maturity T (yr)", ylabel=r"$w=\sigma^2 T$")
    axc.legend(fontsize=8)
    fig_a.tight_layout()
    fig_a.savefig("arbitrage_checks.png", dpi=130)
    print("Saved arbitrage audit -> arbitrage_checks.png")

    # --- Per-tenor smile fits --------------------------------------------- #
    n = len(tenors)
    fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 3.6), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, T in zip(axes, tenors):
        sub = df[df["T"] == T].sort_values("strike")
        F = sub["forward"].iloc[0]
        kk = np.linspace(sub["strike"].min(), sub["strike"].max(), 100)
        p = params[T]
        ax.scatter(sub["strike"] / F, sub["iv"], s=14, color="#d62728",
                   label="market", zorder=3)
        ax.plot(kk / F, hagan_lognormal_vol(F, kk, T, p["alpha"], p["beta"],
                p["rho"], p["nu"]), color="#1f77b4", label="SABR")
        ax.set(title=f"T = {T:.2f}y", xlabel="moneyness K/F")
    axes[0].set_ylabel("implied vol")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig("sabr_smiles.png", dpi=130)
    print("\nSaved per-tenor fits -> sabr_smiles.png")

    # --- Continuous 3D surface via interpolated SABR params --------------- #
    # Interpolating the (smooth, low-dimensional) SABR parameters across tenor
    # yields a continuous surface far more stable than interpolating raw vols.
    moneyness = np.linspace(0.72, 1.28, 60)
    t_fine = np.linspace(tenors.min(), tenors.max(), 40)
    a = np.interp(t_fine, tenors, [params[T]["alpha"] for T in tenors])
    rho = np.interp(t_fine, tenors, [params[T]["rho"] for T in tenors])
    nu = np.interp(t_fine, tenors, [params[T]["nu"] for T in tenors])

    surface = np.empty((t_fine.size, moneyness.size))
    for i, T in enumerate(t_fine):
        F = spot * np.exp((r - q) * T)
        surface[i] = hagan_lognormal_vol(F, moneyness * F, T, a[i], 0.8,
                                         rho[i], nu[i])

    mesh_m, mesh_t = np.meshgrid(moneyness, t_fine)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(mesh_m, mesh_t, surface, cmap="viridis", alpha=0.85,
                    linewidth=0, antialiased=True)
    ax.scatter(df["strike"] / df["forward"], df["T"], df["iv"],
               color="#d62728", s=8, depthshade=True)
    ax.set(xlabel="moneyness K/F", ylabel="maturity T (yr)",
           zlabel="implied vol", title=f"SABR implied-vol surface — {label}")
    ax.view_init(elev=22, azim=-128)
    fig.tight_layout()
    fig.savefig("vol_surface.png", dpi=130)
    print("Saved 3D surface     -> vol_surface.png")


if __name__ == "__main__":
    main()
