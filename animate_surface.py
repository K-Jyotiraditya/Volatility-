"""Interactive, animated 3-D implied-volatility surface (Plotly -> standalone HTML).

This takes a *single* real option-chain snapshot (via yfinance) and animates it
by sweeping a highlighted smile through the listed expiries. The full fitted
SABR surface stays on screen — rotate, zoom, and hover it freely — while the
crimson curve and the black market quotes march from the front tenor out to the
back, so you watch the skew flatten as maturity grows.

A note on honesty: yfinance only exposes the *current* chain, so this animates
the **term structure** of one live snapshot, not a replay of past trading days
(no public feed provides historical surfaces). When the feed is unavailable the
script falls back to the reproducible synthetic market so it always renders.

Everything is built from this project's own modules — no cross-project imports.

    python animate_surface.py            # default ticker SPY, else synthetic
    python animate_surface.py AAPL       # any optionable ticker
"""
from __future__ import annotations

import sys

import numpy as np
import plotly.graph_objects as go

from implied_vol import implied_vol
from market_data import synthetic_chain, load_yfinance_chain
from sabr import calibrate_sabr, hagan_lognormal_vol


def prepare_surface_data(df, spot, r, q, beta=0.8, min_points=6):
    """Invert each quote to an implied vol and fit one SABR smile per expiry.

    Returns a list of per-tenor dicts (sorted by maturity), each carrying the
    market strikes/vols and the calibrated SABR parameters.
    """
    df = df.copy()
    df["iv"] = [implied_vol(row.price, spot, row.strike, r, row.T, row.type, q)
                for row in df.itertuples()]
    df = df.dropna(subset=["iv"])
    df = df[(df["iv"] > 0.03) & (df["iv"] < 2.0)]   # drop illiquid wing garbage

    per_tenor = []
    for T in np.sort(df["T"].unique()):
        sub = df[df["T"] == T].sort_values("strike")
        if len(sub) < min_points:                    # too sparse to fit a smile
            continue
        F = float(sub["forward"].iloc[0])
        strikes = sub["strike"].to_numpy()
        ivs = sub["iv"].to_numpy()
        fit = calibrate_sabr(F, T, strikes, ivs, beta=beta)
        per_tenor.append({"T": float(T), "F": F, "strikes": strikes,
                          "iv": ivs, **fit})
    return per_tenor


def build_animated_surface(per_tenor, spot, r, q, label,
                           out="vol_surface_animated.html",
                           m_lo=0.75, m_hi=1.25, n_m=60):
    """Assemble the Plotly figure and write a self-contained HTML file."""
    tenors = np.array([d["T"] for d in per_tenor])
    beta = per_tenor[0]["beta"]
    # Work in moneyness K/F so every expiry shares one x-axis and the surface is
    # rectangular; absolute strikes drift with the forward and would not align.
    moneyness = np.linspace(m_lo, m_hi, n_m)

    # Smooth surface: interpolate the (low-dimensional) SABR params across tenor
    # rather than the raw vols, then re-evaluate Hagan on a fine maturity grid.
    a = np.array([d["alpha"] for d in per_tenor])
    rho = np.array([d["rho"] for d in per_tenor])
    nu = np.array([d["nu"] for d in per_tenor])
    t_fine = np.linspace(tenors.min(), tenors.max(), 50)
    ai, ri, ni = (np.interp(t_fine, tenors, x) for x in (a, rho, nu))

    surf = np.empty((t_fine.size, n_m))
    for i, T in enumerate(t_fine):
        F = spot * np.exp((r - q) * T)
        surf[i] = hagan_lognormal_vol(F, moneyness * F, T, ai[i], beta, ri[i], ni[i]) * 100

    surface = go.Surface(x=moneyness, y=t_fine, z=surf, colorscale="Viridis",
                         opacity=0.82, colorbar=dict(title="IV %"),
                         name="SABR surface")

    def slice_traces(d):
        """The highlighted smile (line) and market quotes (points) at one expiry."""
        F, T = d["F"], d["T"]
        smile = hagan_lognormal_vol(F, moneyness * F, T, d["alpha"], beta,
                                    d["rho"], d["nu"]) * 100
        line = go.Scatter3d(x=moneyness, y=np.full(n_m, T), z=smile, mode="lines",
                            line=dict(width=7, color="crimson"), name="current smile")
        mk = d["strikes"] / F
        pts = go.Scatter3d(x=mk, y=np.full(mk.size, T), z=d["iv"] * 100,
                           mode="markers", marker=dict(size=3, color="black"),
                           name="market quotes")
        return line, pts

    line0, pts0 = slice_traces(per_tenor[0])

    # Frames update only traces 1 and 2 (the slice); the surface (trace 0) is
    # left untouched, so it stays put as a stable backdrop while the smile moves.
    frames, steps = [], []
    for k, d in enumerate(per_tenor):
        ln, pt = slice_traces(d)
        frames.append(go.Frame(name=str(k), data=[ln, pt], traces=[1, 2]))
        steps.append(dict(method="animate", label=f"{d['T']:.2f}y",
                          args=[[str(k)], dict(mode="immediate",
                                frame=dict(duration=0, redraw=True),
                                transition=dict(duration=0))]))

    fig = go.Figure(data=[surface, line0, pts0], frames=frames)
    fig.update_layout(
        title=f"Implied-vol surface — animated across expiries ({label})",
        scene=dict(xaxis_title="moneyness K / F", yaxis_title="maturity T (yr)",
                   zaxis_title="implied vol (%)",
                   zaxis=dict(range=[max(0.0, surf.min() - 2), surf.max() + 2]),
                   camera=dict(eye=dict(x=1.6, y=-1.7, z=0.8))),
        updatemenus=[dict(type="buttons", x=0.02, y=0.92, showactive=False,
            buttons=[
                dict(label="Play", method="animate",
                     args=[None, dict(frame=dict(duration=700, redraw=True),
                           fromcurrent=True,
                           transition=dict(duration=300, easing="cubic-in-out"))]),
                dict(label="Pause", method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                           mode="immediate")]),
            ])],
        sliders=[dict(active=0, currentvalue=dict(prefix="Front expiry: "),
                      pad=dict(t=60), steps=steps)],
        margin=dict(l=0, r=0, t=50, b=0),
    )
    # Embed plotly.js inline so the file is fully portable and works offline.
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    return out


def main():
    r, q = 0.03, 0.0
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "SPY"
    try:
        df = load_yfinance_chain(ticker, r, q, max_expiries=12)
        spot, label = df.attrs["spot"], f"{ticker} live snapshot"
        print(f"Loaded live chain for {ticker}: spot {spot:.2f}, {len(df)} quotes")
    except Exception as exc:
        print(f"[warn] live fetch failed ({exc}); using synthetic market.")
        df = synthetic_chain(r=r, q=q,
                             tenors=(0.05, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0))
        spot, label = 100.0, "synthetic"

    per_tenor = prepare_surface_data(df, spot, r, q)
    if len(per_tenor) < 2:
        raise SystemExit("Not enough usable expiries to animate a surface.")
    out = build_animated_surface(per_tenor, spot, r, q, label)
    print(f"Animated {len(per_tenor)} expiries -> {out}")
    print("Open it in any browser, then press Play (or scrub the slider).")


if __name__ == "__main__":
    main()
