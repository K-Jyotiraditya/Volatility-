# Project 2 — Theory & Deep Dive: Implied Volatility Surface & SABR

How a desk turns a screen full of option quotes into a smooth, arbitrage-free
volatility surface — and why every step is harder than it looks. Maps directly
to `black_scholes.py`, `implied_vol.py`, `sabr.py`, and `arbitrage.py`.

---

## 1. The one number that breaks Black-Scholes

Black-Scholes assumes a single constant volatility `σ`. Invert the formula —
solve for the `σ` that reproduces each market price — and you get the
**implied volatility**. If the model were true, implied vol would be identical
across all strikes and maturities: a flat sheet. It is not. Plot it and you see
a **smile** (or **skew**): out-of-the-money puts trade at higher implied vol
than calls. This is the market pricing in fat tails and the leverage effect that
lognormal dynamics cannot capture. The implied-vol surface `σ(K, T)` is the
market's own correction to Black-Scholes, and modelling it *is* the job.

Implied vol is the lingua franca of options: quotes, risk, and hedging are all
expressed in vol, not price. A "25-delta risk reversal" or "ATM term structure"
are statements about this surface.

---

## 2. Inverting Black-Scholes — robust root-finding

### 2.1 The problem

Given a market price `C*`, find `σ` such that `C_BS(σ) = C*`. Since `C_BS` is
**strictly increasing** in `σ` (vega > 0), the root is unique when it exists. It
exists only if `C*` lies inside the no-arbitrage bounds
`[max(S e^{−qT} − K e^{−rT}, 0), S e^{−qT}]`; outside, the code returns `NaN`
rather than chase a phantom.

### 2.2 Why naive Newton blows up — and the fix

Newton's update is `σ ← σ − f(σ)/vega`. In the deep wings vega → 0, so the step
explodes and launches `σ` to absurd values. The engine uses the **`rtsafe`
hybrid**: maintain a bracket `[lo, hi]` that always contains the root, take a
Newton step when it stays inside, and **bisect** whenever it would escape.
Newton's quadratic speed with bisection's guaranteed convergence.

The subtle bug the code avoids: **convergence is judged on the step in `σ`, not
on an absolute price residual.** A deep-wing option can be worth `~1e-10`; a
price tolerance of `1e-8` would "converge" immediately at the initial guess,
returning a totally wrong vol. This single design choice is the difference
between a solver that works on the liquid ATM strikes and one that survives the
wings where the smile is most informative.

Brent's method is offered as an alternative — slightly slower but derivative-free
and bulletproof.

---

## 3. SABR — a model for the smile

### 3.1 The dynamics

SABR (Stochastic Alpha-Beta-Rho; Hagan, Kumar, Lesniewski, Woodward 2002) models
the *forward* and its volatility as coupled diffusions:

```
dF = α F^β dW,    dα = ν α dZ,    d⟨W, Z⟩ = ρ dt.
```

Four parameters, each with clean intuition:

| Param | Name | Effect on the smile |
|-------|------|---------------------|
| `α` | initial vol level | sets the **height** (ATM level) |
| `β` | backbone elasticity | `1`=lognormal, `0`=normal; governs the ATM/spot dynamics |
| `ρ` | spot-vol correlation | **tilts** the smile into a skew (negative ρ ⇒ downside skew) |
| `ν` | vol-of-vol | sets the **curvature** (how fast vol rises in the wings) |

### 3.2 Hagan's formula and its singular points

The headline result is a *closed-form asymptotic expansion* for the Black
implied vol as a function of strike — no simulation needed to evaluate the
smile. The implementation carefully handles two **removable singularities**:

- **At the money** (`F = K`): the `z/x(z)` factor is `0/0`; the limit is `1`.
- **Backbone series** in `ln(F/K)`: handled by the explicit expansion terms.

Computed under `np.errstate` because those entries are genuine `0/0` that we
immediately overwrite with the analytic limit.

### 3.3 Why we fix β

`β` and `ρ` are **jointly under-identified**: a change in `β` can be largely
undone by a change in `ρ` while barely moving the fit. So `β` is fixed by asset
class (≈1 for equity/FX, ≈0.5 for rates) and only `(α, ρ, ν)` are calibrated.
Trying to fit all four to one tenor's smile gives unstable, meaningless numbers.

### 3.4 Calibration

For each maturity we minimise the squared vol error by nonlinear least squares
(`scipy.optimize.least_squares`, trust-region). On the synthetic market — built
from a *known* SABR term structure — the calibrator recovers the generating
parameters to ~1e-3 and the IV solver matches ground-truth vols to quote-noise
level. That closed validation loop is something live quotes can never give you.

### 3.5 SVI — the alternative

The Stochastic Volatility Inspired (SVI) parameterisation models *total
variance* directly: `w(k) = a + b(ρ(k−m) + √((k−m)² + σ²))`. It is a pure fit
(no dynamics) but has the advantage that arbitrage-free conditions can be
imposed analytically (Gatheral-Jacquier "SSVI"). SABR is chosen here because its
parameters carry dynamic meaning and it is the rates/FX desk standard.

---

## 4. Static arbitrage — a smooth surface is not enough

A fitted surface can be beautifully smooth and still let someone print money.
Two **static** (single-snapshot) conditions must hold.

### 4.1 No butterfly arbitrage (convexity in strike)

By the **Breeden-Litzenberger** identity, the risk-neutral density of the
terminal price is the second derivative of the call price in strike:

```
g(K) = e^{rT} · ∂²C/∂K²    (must be ≥ 0, since it is a probability density).
```

A negative density means a **butterfly spread** (long the `K−ΔK` and `K+ΔK`
calls, short two `K` calls) costs less than zero while paying off ≥ 0 — free
money. `arbitrage.py` prices the SABR smile into calls (Black-76), takes the
second difference, and flags any region where `g < 0`. Hagan's expansion is
known to violate this in the **far wings at short maturity with high vol-of-vol**
— exactly the regime the test deliberately triggers (a density of −0.0024 at
`T=0.25, ν=6, ρ=−0.7`).

### 4.2 No calendar arbitrage (monotone total variance)

Define **total implied variance** `w(k, T) = σ²(k, T)·T` at fixed
forward-log-moneyness `k = ln(K/F)`. No-calendar-arbitrage requires

```
∂w/∂T ≥ 0    for every k.
```

Intuitively, a longer-dated option must contain at least as much uncertainty as
a shorter-dated one at the same moneyness; if not, a calendar spread is riskless
profit. `calendar_report` checks the increments of `w` across the tenor grid.

### 4.3 Penalised calibration

`calibrate_sabr_arbfree` augments the least-squares residual with
`penalty · max(−g, 0)` sampled across strikes, steering the optimiser away from
parameter sets whose wings imply a negative density — a *soft* constraint that
keeps the fit stable rather than a hard constraint that can make it brittle. On a
clean smile the penalty is inactive and the fit is unchanged.

---

## 5. Building the surface

We do **not** interpolate raw implied vols across maturity — that invites
calendar arbitrage and jagged forwards. Instead we interpolate the **SABR
parameters** `(α, ρ, ν)` (smooth, low-dimensional, economically meaningful)
across tenor and re-evaluate Hagan on a fine moneyness × maturity grid. The
result is continuous by construction and far more stable.

The animated view (`animate_surface.py`) sweeps a highlighted smile through the
listed expiries, so you watch the skew flatten with maturity — the term
structure of skew made visual.

---

## 6. Where this lives in the real world

- **Market making:** quote any strike/tenor consistently from a handful of
  parameters; the surface *is* the inventory of risk.
- **Exotics pricing:** the smile is the input to local-vol (Dupire) and
  stochastic-vol models; a wrong surface means wrong barrier/cliquet prices.
- **Risk management:** vega, vanna, volga are sensitivities *to the surface*;
  scenario P&L shocks the surface, not a single vol.
- **Arbitrage policing:** desks reject or repair arbitrageable marks before they
  feed downstream pricing — precisely the audit implemented here.

---

## 7. Limitations and extensions

- **Hagan wing arbitrage.** The asymptotic formula is not guaranteed
  arbitrage-free; production desks use the no-arbitrage SABR PDE (Hagan 2014) or
  SSVI in the wings. The penalty here mitigates rather than eliminates this.
- **Forward construction.** We use `F = S·e^{(r−q)T}`; a real desk implies the
  forward from put-call parity to absorb dividends and funding precisely.
- **Quote hygiene.** Live chains need filtering for stale quotes, wide/crossed
  markets, and weighting by liquidity (vega or open-interest weights).
- **Joint calibration.** Fitting the whole surface at once with calendar
  constraints (rather than tenor-by-tenor) gives a globally consistent fit.
