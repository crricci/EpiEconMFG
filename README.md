# EpiEconMFG

Julia code for an epidemiological-economic Mean Field Game with heterogeneous
agents over wealth/capital and epidemiological states.

The project contains two solvers:

1. A fully dynamic forward-backward solver, where both the HJB system and the
   Fokker-Planck equation are time dependent. This is the main model.
2. A quasi-static solver, where the Fokker-Planck equation evolves forward in
   time while a stationary HJB is re-solved at each date. This is not the full
   dynamic solution, but it is useful as a benchmark and as the default initial
   guess for the dynamic solver.

The current model includes an exogenous monetary vaccination cost `ξ(t,k)`.
In the baseline calibration this function is constant and controlled by `p.ξ`.

## Quick Start

From the shell:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); include("main.jl"); result = run_dynamic()'
```

From the Julia REPL:

```julia
include("main.jl")

p = EpiEconMFG.MFGEpiEcon()
F0 = EpiEconMFG.create_test_distribution(p)

result = run_dynamic(p = p, F0 = F0, show_progress = true)
```

To run the quasi-static solver instead:

```julia
include("main.jl")

p = EpiEconMFG.MFGEpiEcon()
F0 = EpiEconMFG.create_test_distribution(p)

result_qs = run(p = p, F0 = F0, show_progress = true)
```

## Repository Structure

```text
main.jl
    Interactive entry points:
    - run() for the quasi-static solver
    - run_dynamic() for the fully dynamic solver

src/EpiEconMFG.jl
    Main module, includes, and exports.

src/core/parameters.jl
    Model parameters, grids, prices, initial distribution, and ξ(t,k).

src/core/aggregates.jl
    Labor policies, aggregate capital, aggregate labor, infected labor,
    wages, and interest rates.

src/core/diff.jl
    Finite differences on the capital grid.

src/solvers/hjb_time_dependent.jl
    Backward time-dependent HJB solver.

src/solvers/coupled_forward_backward.jl
    Full dynamic forward-backward fixed point.

src/solvers/hjb_stationary.jl
    Stationary HJB solver used by the quasi-static method.

src/solvers/fp_kfe.jl
    Fokker-Planck/Kolmogorov forward equation, policy construction,
    and distribution dynamics.

src/solvers/coupled_quasistatic.jl
    Quasi-static coupled solver.

src/visualization/plots.jl
    Plotting utilities.

scripts/
    Diagnostics and exploratory scripts.

docs/
    Development notes.

outputs/
    Generated outputs.
```

## Model

Agents are heterogeneous in capital `k` and in epidemiological state

```text
S = susceptible
I = infected
C = contained
R = recovered
```

Let

```math
\phi_e(t,k), \qquad e\in\{S,I,C,R\},
```

denote the distribution of agents in state `e`.

The individual controls are consumption and labor,

```math
c_e(t,k)\ge 0,
\qquad
l_e(t,k)\in[0,1],
```

and susceptible agents additionally choose a vaccination intensity

```math
q(t,k)\ge 0.
```

Flow utility is

```math
u(c,l)=\theta\log(c)+(1-\theta)\log(1-l).
```

Infected and contained agents receive health disutility terms `dI` and `dC`.
Vaccination has a quadratic utility cost and a monetary cost in the household
budget.

## Aggregates and Prices

The infected labor externality is

```math
L_I(t)=\int l_I(t,k)\phi_I(t,k)\,dk.
```

Aggregate capital is

```math
K(t)=
\sum_{e\in\{S,I,C,R\}}
\int k\phi_e(t,k)\,dk.
```

Effective aggregate labor is

```math
L(t)=
\sum_{e\in\{S,I,C,R\}}
\eta_e\int l_e(t,k)\phi_e(t,k)\,dk.
```

Production is Cobb-Douglas:

```math
Y(t)=A K(t)^\alpha L(t)^{1-\alpha}.
```

The competitive prices are

```math
r(t)=\alpha A K(t)^{\alpha-1}L(t)^{1-\alpha},
```

```math
w(t)=(1-\alpha)A K(t)^\alpha L(t)^{-\alpha}.
```

## Fully Dynamic Problem

The fully dynamic model is a forward-backward system:

- the HJB system is solved backward in time;
- the Fokker-Planck equation is solved forward in time;
- prices and aggregate infection exposure are determined by the evolving
  distribution and policies.

### Household Objective

For an individual household, the finite-horizon problem is

```math
\max_{c,l,q}
\mathbb{E}
\left[
\int_0^T e^{-\rho t}
\left(
u(c_t,l_t)
- d_I\mathbf{1}_{\{e_t=I\}}
- d_C\mathbf{1}_{\{e_t=C\}}
- \frac{\gamma}{2}q_t^2\mathbf{1}_{\{e_t=S\}}
\right)dt
+ e^{-\rho T}V_T(e_T,k_T)
\right].
```

The monetary vaccination cost enters the susceptible budget constraint:

```math
\dot{k}_S
=
(r(t)-\delta)k
+ w(t)\eta_S l_S
- c_S
- \xi(t,k)q.
```

The other capital drifts are

```math
\dot{k}_I
=
(r(t)-\delta)k
+ w(t)\eta_I l_I
- c_I,
```

```math
\dot{k}_C
=
(r(t)-\delta)k
- c_C,
```

```math
\dot{k}_R
=
(r(t)-\delta)k
+ w(t)\eta_R l_R
- c_R.
```

### Dynamic HJB System

For susceptible agents:

```math
\rho V_S(t,k)
=
\partial_t V_S(t,k)
+ \max_{c\ge0,\;l\in[0,1],\;q\ge0}
\Big\{
u(c,l)
- \frac{\gamma}{2}q^2
+ \partial_k V_S(t,k)
\big[
(r(t)-\delta)k+w(t)\eta_S l-c-\xi(t,k)q
\big]
```

```math
\qquad
+ q\big[V_R(t,k)-V_S(t,k)\big]
+ \beta l L_I(t)\big[V_I(t,k)-V_S(t,k)\big]
\Big\}.
```

For infected agents:

```math
\rho V_I(t,k)
=
\partial_t V_I(t,k)
+ \max_{c\ge0,\;l\in[0,1]}
\Big\{
u(c,l)-d_I
+ \partial_k V_I(t,k)
\big[(r(t)-\delta)k+w(t)\eta_I l-c\big]
```

```math
\qquad
+ \mu\big[V_S(t,k)-V_I(t,k)\big]
+ \sigma_1\big[V_C(t,k)-V_I(t,k)\big]
+ \sigma_3\big[V_R(t,k)-V_I(t,k)\big]
\Big\}.
```

For contained agents:

```math
\rho V_C(t,k)
=
\partial_t V_C(t,k)
+ \max_{c\ge0}
\Big\{
\theta\log(c)-d_C
+ \partial_k V_C(t,k)\big[(r(t)-\delta)k-c\big]
```

```math
\qquad
+(\alpha_{Epi}+\mu)\big[V_S(t,k)-V_C(t,k)\big]
+\sigma_2\big[V_R(t,k)-V_C(t,k)\big]
\Big\}.
```

For recovered agents:

```math
\rho V_R(t,k)
=
\partial_t V_R(t,k)
+ \max_{c\ge0,\;l\in[0,1]}
\Big\{
u(c,l)
+ \partial_k V_R(t,k)
\big[(r(t)-\delta)k+w(t)\eta_R l-c\big]
```

```math
\qquad
+(\lambda+\mu)\big[V_S(t,k)-V_R(t,k)\big]
\Big\}.
```

The vaccination first-order condition gives

```math
q^*(t,k)
=
\frac{V_R(t,k)-V_S(t,k)-\xi(t,k)\partial_k V_S(t,k)}{\gamma},
```

with non-negativity and upper-bound clipping in the implementation.

### Fokker-Planck System

Let

```math
\nu(t,k)=\beta l_S(t,k)L_I(t)
```

be the infection intensity faced by susceptible agents. The distribution evolves
according to

```math
\partial_t\phi_S
=
-\partial_k(\phi_S b_S)
-(\nu+q)\phi_S
+\mu\phi_I
+(\alpha_{Epi}+\mu)\phi_C
+(\lambda+\mu)\phi_R,
```

```math
\partial_t\phi_I
=
-\partial_k(\phi_I b_I)
+\nu\phi_S
-(\sigma_1+\sigma_3+\mu)\phi_I,
```

```math
\partial_t\phi_C
=
-\partial_k(\phi_C b_C)
+\sigma_1\phi_I
-(\alpha_{Epi}+\sigma_2+\mu)\phi_C,
```

```math
\partial_t\phi_R
=
-\partial_k(\phi_R b_R)
+q\phi_S
+\sigma_3\phi_I
+\sigma_2\phi_C
-(\lambda+\mu)\phi_R.
```

For susceptible agents,

```math
b_S(t,k)
=
(r(t)-\delta)k
+w(t)\eta_S l_S(t,k)
-c_S(t,k)
-\xi(t,k)q(t,k).
```

## Dynamic Fixed Point

The main dynamic solver is `solveModelDynamic`, exposed through `run_dynamic`.
It iterates on the full time path of distributions, values, controls, and prices.

```text
Given parameters p and initial distribution F0:

1. Build the time grid.

2. Initialize paths for F, V, and controls.
   By default this is done with the quasi-static solver.

3. Fix the terminal value function VT.

4. Repeat until convergence:

   a. Compute aggregate paths and price paths from the current paths.

   b. Solve the time-dependent HJB backward in time,
      taking the current distribution and price paths as given.

   c. Reconstruct policies from the new value path.

   d. Solve the Fokker-Planck equation forward in time from F0.

   e. Recompute prices and aggregates.

   f. Check convergence in distributions, values, wages, and interest rates.

   g. If not converged, damp the distribution and value paths and iterate.
```

The convergence criterion is based on

```math
\max\{err_F,err_V,err_w,err_r\}.
```

The dynamic solver returns the time grid, distribution path, value path,
controls, prices, aggregates, diagnostics, convergence flag, and number of
iterations.

## Quasi-Static Solver

The quasi-static method solves a time-dependent Fokker-Planck equation while
replacing the dynamic HJB with a stationary HJB at each date.

Given the current distribution `F(t)`, the stationary HJB has the form

```math
\rho V_e(k)
=
\max_{\text{controls}}
\left\{
\text{flow payoff}
+b_e(k)\partial_k V_e(k)
+\text{epidemiological transitions at the current date}
\right\}.
```

The key difference from the fully dynamic problem is that the quasi-static HJB
does not include

```math
\partial_t V_e(t,k).
```

It therefore does not internalize the full future path of prices, distributions,
infection risk, and vaccination incentives.

### Quasi-Static Fixed Point

At each date, the stationary HJB is solved with a nested fixed point over wages
and value functions:

```text
Given the current distribution Ft:

1. Start from a wage guess.

2. Solve the stationary HJB at the current wage by value iteration.

3. Compute aggregate labor and capital from the resulting policies.

4. Compute the implied wage from the production function.

5. Update the wage with damping.

6. Repeat until the wage fixed point converges.
```

The outer quasi-static loop then advances the Fokker-Planck equation one time
step using the policies from the stationary HJB:

```text
Given F0:

for each time step:
    solve stationary HJB and wage fixed point at current F
    construct policies and the FP generator
    advance the distribution forward one step
    save values, controls, prices, and distributions
```

## Dynamic Initialization

By default, the dynamic solver uses

```julia
dynamicInitialGuess = :quasistatic
```

This means that the quasi-static solver is run first. Its paths for
distributions, value functions, and controls are then used as the initial guess
for the full forward-backward dynamic fixed point.

This initialization is useful because the quasi-static path already satisfies
the forward distribution dynamics and the static optimality restrictions, even
though it is not fully forward-looking.

## Main Output

`solveModelDynamic` returns a `NamedTuple` with:

```text
t
    Saved time grid.

F
    Distribution path.

V
    Value-function path, including associated w, r, and LI.

controls
    Consumption, labor, vaccination, drift, and transition intensities.

prices
    Wage, interest rate, aggregate capital, aggregate labor, and infected labor.

aggregates
    Aggregate capital, aggregate labor, and infected labor.

diagnostics
    Fixed-point residuals and FP/HJB diagnostics.

converged
    Convergence flag.

iterations
    Number of dynamic Picard iterations.

method
    :forward_backward_dynamic
```

## Current Status

The fully dynamic solver is implemented in `solveModelDynamic`.

The quasi-static solver remains in the repository because it is useful for:

- generating the default initial guess for the dynamic solver;
- comparing the full dynamic solution with a non-forward-looking benchmark;
- debugging the HJB and Fokker-Planck components separately.
