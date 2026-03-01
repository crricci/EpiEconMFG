# EpiEconMFG

Julia code for an epidemic-economic Mean Field Game (MFG) with heterogeneous agents over capital and epidemiological states.

The code solves:
- A stationary-in-form HJB system (re-solved over time because aggregates and infection depend on the current distribution).
- A forward Kolmogorov / Fokker-Planck (FP/KFE) equation for the joint distribution over capital and health states.
- A coupled numerical scheme with FP time-marching as the outer loop, a wage fixed point as an outer stationary loop, and HJB value iteration as the inner stationary loop.

## Code structure and code usage

### Repository structure

```text
L_parameters.jl         # Parameters and initial distribution
L_diff.jl               # Safe first derivative on capital grid
L_aggregateVariables.jl # Optimal labor and aggregate K, L
L_wageSolver.jl         # Wage fixed-point map (plus robust root fallback)
L_HJBsolver.jl          # HJB linear-system assembly + value iteration + wage outer loop
L_KFEsolver.jl          # FP policy extraction + generator assembly + implicit Euler
L_MFGCsolver.jl         # Main coupled solver `solveModel`
L_plots.jl              # Figure generation (heatmaps + surfaces)
L_loadAll.jl            # Dependency loading + includes all code modules
main.jl                 # `run()` entrypoint (solve + save figures)
DEBUG_HJB.jl            # HJB diagnostics / residual checks
DEBUG_FP.jl             # FP diagnostics
DEBUG.jl                # Quick one-shot run
```

### Main data objects

- Distribution at one time: $F_t=(\phi_S(t,\cdot),\phi_I(t,\cdot),\phi_C(t,\cdot),\phi_R(t,\cdot))$, stored as `Ft = (ϕSt, ϕIt, ϕCt, ϕRt)` (each vector length `Nk`).
- Value functions: `V = (VS, VI, VC, VR)`, each vector length `Nk`.
- Coupled output from `solveModel`:
  - `result.t`: saved time grid.
  - `result.F`: saved distributions.
  - `result.V`: saved HJB solutions and wage.
  - `result.controls`: saved controls and drifts used in FP.

### How to run

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); include("main.jl"); run()'
```

Equivalent from REPL:

```julia
include("L_loadAll.jl")

p = MFGEpiEcon()
F0 = create_test_distribution(p)

result = solveModel(p, F0; show_progress=true)
save_all_figures(result, p)
```

### Typical customization

```julia
include("L_loadAll.jl")

p = MFGEpiEcon(
    MaxK = 80.0,
    Δk = 0.5,
    T_End = 3.0,
    Δt = 0.02,
    HJB_every = 1,
    verbose = true
)

F0 = create_test_distribution(p)
result = solveModel(p, F0; show_progress=false, save_stride=2)
```

If $\mathrm{HJB\_every}>1$, controls are frozen between HJB updates for speed.

## Math problem that we are solving

### States and controls

- Epidemiological states: `S` (susceptible), `I` (infected), `C` (contained), `R` (recovered).
- Individual state variable: capital $k\ge 0$.
- Controls: consumption $c$, labor $l$ (with $l_C=0$ in implementation), and vaccination intensity $q$ for susceptible agents.

### Forward equation (FP/KFE)

Let $\phi_e(t,k)$ be the cross-sectional density in state $e$.

For `S`:

```math
\partial_t \phi_S
= \mu(\phi_S+\phi_I+\phi_C+\phi_R)
+ \alpha_E\phi_C
- \mu\phi_S
- q^*(t,k)\phi_S
- \beta l_S^*(t,k)\phi_S L_I
+ \lambda\phi_R
- \partial_k\!\left(\phi_S b_S\right),
```

where
```math
b_S=(r_t-\delta)k + w_t\eta(S)l_S^*(t,k)-c_S^*(t,k).
```

For `I`:

```math
\partial_t \phi_I
= -(\sigma_1+\mu+\sigma_3)\phi_I
+ \beta l_S^*(t,k)\phi_S L_I
- \partial_k\!\left(\phi_I b_I\right),
```
```math
b_I=(r_t-\delta)k + w_t\eta(I)l_I^*(t,k)-c_I^*(t,k).
```

For `C`:

```math
\partial_t \phi_C
= \sigma_1\phi_I
- (\alpha_E+\sigma_2+\mu)\phi_C
- \partial_k\!\left(\phi_C b_C\right),
```
```math
b_C=(r_t-\delta)k-c_C^*(t,k).
```

For `R`:

```math
\partial_t \phi_R
= \sigma_2\phi_C + \sigma_3\phi_I - (\lambda+\mu)\phi_R + q^*(t,k)\phi_S
- \partial_k\!\left(\phi_R b_R\right),
```
```math
b_R=(r_t-\delta)k + w_t\eta(R)l_R^*(t,k)-c_R^*(t,k).
```

Aggregate infected labor term:

```math
L_I(t)=\int l_I^*(t,k)\phi_I(t,k)\,dk.
```

### Household objective and HJB system

The model solves (state-contingent) dynamic programs with discount $ρ$.
Utility in code is implemented as:

```math
u(c,l)=\theta\log(c)+(1-\theta)\log(1-l),
```

with additional health disutility $d_I$, $d_C$, and vaccination cost $-\frac{\gamma}{2}q^2$.

Representative objective (state-dependent controls and transitions):

```math
\max_{(c,l,q)}\;\mathbb{E}\!\left[\int_0^\infty e^{-\rho t}
\left(
u(c_t, l_t)
- d_I\,\mathbf{1}_{\{e_t=I\}}
- d_C\,\mathbf{1}_{\{e_t=C\}}
- \frac{\gamma}{2}q_t^2
\right)dt\right].
```

HJB for `S`:

```math
\rho V_S(k)=\max_{c\ge0,\,l\in[0,1],\,q\ge0}
\left\{
u(c,l)+V'_S(k)\big[(r-\delta)k+w l-c\big]
+q\big(V_R-V_S\big)
+\beta l L_I \big(V_I-V_S\big)
-\frac{\gamma}{2}q^2
\right\}.
```

HJB for `I`:

```math
\rho V_I(k)=\max_{c\ge0,\,l\in[0,1]}
\left\{
u(c,l)-d_I
+V'_I(k)\big[(r-\delta)k+w\eta(I)l-c\big]
+\sigma_1(V_C-V_I)+\mu(V_S-V_I)+\sigma_3(V_R-V_I)
\right\}.
```

HJB for `C`:

```math
\rho V_C(k)=\max_{c\ge0}
\left\{
\theta\log(c)-d_C
+V'_C(k)\big[(r-\delta)k-c\big]
+(\alpha_E+\mu)(V_S-V_C)+\sigma_2(V_R-V_C)
\right\}.
```

HJB for `R`:

```math
\rho V_R(k)=\max_{c\ge0,\,l\in[0,1]}
\left\{
u(c,l)+V'_R(k)\big[(r-\delta)k+w l-c\big]
+(\lambda+\mu)(V_S-V_R)
\right\}.
```

### Aggregates and prices

```math
K_t=\sum_{e\in\{S,I,C,R\}}\int k\,\phi_e(t,k)\,dk,
\quad
L_t=\sum_e \eta(e)\int l_e^*(t,k)\phi_e(t,k)\,dk.
```

With production:

```math
Y_t=A K_t^\alpha L_t^{1-\alpha},
\quad
r_t=\alpha A K_t^{\alpha-1}L_t^{1-\alpha},
\quad
w_t=(1-\alpha)A K_t^\alpha L_t^{-\alpha}.
```

### Implemented policy formulas

The code uses these pointwise controls from HJB FOCs:

```math
c_e^*(k)=\frac{\theta}{V'_e(k)},\qquad e\in\{S,I,C,R\},
```

```math
l_e^*(k)=\max\!\left(0,\min\!\left(1,1-\frac{1-\theta}{V'_e(k)W_e(k)}\right)\right),\qquad e\in\{S,I,R\},
```
and $l_C^*=0$ (because $\eta_C=0$, implemented through effective wage clipping).

Effective wages:

```math
W_I=\eta_I w,\quad W_C=\eta_C w,\quad W_R=\eta_R w,
```
```math
W_S(k)=\eta_S w + \beta L_I\frac{V_I(k)-V_S(k)}{V'_S(k)}.
```

Vaccination:

```math
q^*(k)=\min\!\left\{q_{\max},\max\!\left\{0,\frac{V_R(k)-V_S(k)}{\gamma}\right\}\right\}.
```

## Numerical details

### 1. Grids and time discretization

- Capital grid: $k_i=(i-1)\Delta k$, $i=1,\dots,N_k$, with $N_k=\mathrm{Int}(\mathrm{MaxK}/\Delta k)+1$.
- Time step for FP in `solveModel`:
  - $N_{\mathrm{step}}=\lceil T_{\mathrm{End}}/\Delta t\rceil$.
  - Effective step used in simulation: $\Delta t_{\mathrm{eff}}=T_{\mathrm{End}}/N_{\mathrm{step}}$.
- States are stacked as a vector of length $4N_k$ in order $(S,I,C,R)$.

### 2. Safe derivative and control stability

`∂k_safe!` computes a numerical approximation of $\partial_k V_e(k)$ using:
- One-sided differences at boundaries.
- Central differences inside the domain.
- A positivity floor `ϵDkUp` on $V'(k)$.

This avoids division by zero in $c=\theta/V'$ and in labor/effective wage terms.

### 3. HJB discretization and solver

For fixed `w`, HJB is solved by repeated application of:

```math
(\rho I - A - Q)V = u,
```

where:
- $A$ is the upwind advection matrix for $-b_e(k)\partial_k V_e(k)$.
- $Q$ is the health-state transition generator.
- $u$ is the flow utility vector.

Implementation specifics:
- Upwind split at each grid point: $b^+=\max(b,0)$, $b^-=\min(b,0)$.
- Sparse matrix assembled by triplets `(I,J,X)` then solved with sparse backslash.
- Value iteration damping:
  - $V\leftarrow (1-\omega)V+\omega T(V)$.
- Convergence criterion:
  - $\max_e\|T(V_e)-V_e\|_\infty<\mathrm{tolHJBvalue}$.

### 4. Boundary state constraints in capital

The code enforces state constraints at $k=0$ and $k=k_{\max}$ (implemented as the first and last grid nodes) in two places:

- On controls (consumption clipping):
  - At $k=0$: force $b_e(0)\ge 0$ by imposing $c_e(0)\le \mathrm{income}_e(0)$.
  - At $k=k_{\max}$: force $b_e(k_{\max})\le 0$ by imposing $c_e(k_{\max})\ge \mathrm{income}_e(k_{\max})$.
- On drift directly:
  - $b_e[1]=\max\{b_e[1],0\}$, $b_e[N_k]=\min\{b_e[N_k],0\}$.

This is consistent with no-outflow state constraints for both HJB and FP operators.

### 5. Wage fixed point

Inside each stationary HJB solve:

1. Start from `w_start`.
2. Solve HJB at current wage.
3. Compute implied wage from aggregates and production function.
4. Update with damping:
   - $w\leftarrow (1-\omega_w)w+\omega_w\,w_{\mathrm{implied}}$.
5. Stop when $|w_{\mathrm{implied}}-w|<\mathrm{tolWage}$.

### 6. FP/KFE generator and time stepping

Given current controls, FP uses:

```math
\dot{\phi}=G\phi.
```

with:
- Conservative upwind drift block per state.
- Local epidemiological transition flows at each capital node.

Time stepping is implicit Euler:

```math
(I-\Delta t_{\mathrm{eff}}G^n)\phi^{n+1}=\phi^n.
```

After each step, numerical safety fixes are applied:
- Project density to nonnegative values.
- Renormalize total mass to 1.

### 7. Coupled algorithm implemented in `solveModel`

The implemented loop is explicitly nested:

1. FP time-marching loop (outermost): for $n=0,\dots,N_t-1$, given $\phi^n$.
2. Stationary equilibrium update at time step $n$ (performed every `HJB_every` steps):
   - Outer fixed point on wage $w$:
     ```math
     w^{m+1}=(1-\omega_w)w^m+\omega_w\,T_w(w^m;V,\phi^n),
     ```
     stopped when $|w^{m+1}-w^m|<\mathrm{tolWage}$.
   - Inner fixed point on HJB for each wage iterate:
     ```math
     V^{j+1}=(1-\omega)V^j+\omega\,T_{\mathrm{HJB}}(V^j;w^m,\phi^n),
     ```
     stopped when $\|V^{j+1}-V^j\|_\infty<\mathrm{tolHJBvalue}$.
3. With converged $(V^n,w^n)$, build controls and generator $G^n$.
4. FP one-step time march (Backward Euler):
   ```math
   (I-\Delta t_{\mathrm{eff}}G^n)\phi^{n+1}=\phi^n.
   ```
5. Project to nonnegative mass, renormalize to total mass $1$, and save outputs.

$\mathrm{HJB\_every}=1$ is fully coupled; larger values trade accuracy for speed.

### 8. Diagnostics and plotting

- `DEBUG_HJB.jl` checks:
  - HJB fixed-point residual.
  - Linear-system residual $\|MV-\mathrm{rhs}\|_\infty$.
  - Wage residual.
- `DEBUG_FP.jl` checks:
  - Distribution mass preservation.
  - Infected mass change.
  - Final prices and aggregates.
- `save_all_figures(result,p)` produces:
  - Compartments over time.
  - Density heatmaps/surfaces.
  - Infection/vaccination flows.
  - Control heatmaps/surfaces (`c`, `l`, `q`).
