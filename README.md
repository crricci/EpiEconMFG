# EpiEconMFG: Epidemic-Economic Mean Field Game Solver

A Julia implementation of a Mean Field Game (MFG) framework for modeling epidemic-economic dynamics with heterogeneous agents and capital accumulation.

## Overview

This project solves a system of stationary Hamilton-Jacobi-Bellman (HJB) equations on a capital grid and a (time-dependent) forward Kolmogorov / Fokker--Planck (FP) equation for the cross-sectional distribution.

Numerically, the distribution is advanced forward in time, while at each distribution update (or every few steps) the code recomputes the stationary HJB policies and the competitive wage fixed point implied by the current distribution.

### Epidemiological States

The model tracks four epidemiological states:
- **S (Susceptible)**: Healthy agents at risk of infection
- **I (Infected)**: Asymptomatic infected agents  
- **C (Contained)**: Symptomatic/quarantined agents
- **R (Recovered)**: Immune agents

### Key Features

- **Capital Accumulation**: Agents accumulate capital with state-dependent productivity
- **Labor Supply Choice**: Endogenous labor-leisure decision with participation constraints
- **Vaccination Decision**: Susceptible agents choose vaccination propensity
- **Infection Risk**: Infection intensity depends on population distribution
- **Stationary HJB Solve**: Policies from pointwise FOCs + implicit sparse linear solve on the grid
- **Upwind Finite Differences**: Upwind discretization for the capital drift term

## Model Structure

### Stationary HJB System (Implemented)

Let $x \in \{S,I,C,R\}$ denote the epidemiological state and $k \in [0, k_{max}]$ the individual capital state (discretized on a grid).

The code solves the stationary HJB system

$$\rho V_x(k)  = u_x(k) + b_x(k) \partial_k V_x(k) + \sum_{y\neq x} q_{xy}(k) \big(V_y(k)-V_x(k)\big),\quad x\in\{S,I,C,R\},$$

with capital drift

$$ b_x(k) = (r-\delta)k + \eta_x w \ell_x(k) - c_x(k) \quad (x\in\{S,I,R\}),\qquad b_C(k) = (r-\delta)k - c_C(k).$$

`Ft = (ϕSt, ϕIt, ϕCt, ϕRt)` is the cross-sectional distribution over states and the $k$-grid. It is used to compute aggregates and the infection externality, and it can either be treated as an exogenous input (pure stationary solve) or evolved endogenously via the FP equation (dynamic distribution).

#### Controls

Consumption is taken from the FOC 

$$c_x(k) = \frac{\theta}{\partial_k V_x(k)}.$$

Labor is chosen by the closed-form rule

$$\ell_x(k) = \max\left(0,1 - \frac{1-\theta}{\partial_k V_x(k)W_x(k)}\right),$$
where effective wages are
$$
W_I = \eta_I w,\quad W_R = \eta_R w,\quad W_C = 0,
$$
and for susceptibles (capturing the infection externality wedge)
$$
W_S(k) = \eta_S w + \beta LI\frac{V_I(k)-V_S(k)}{\partial_k V_S(k)},
\qquad LI = \int \ell_I(k) \phi_I(k) dk.
$$

Vaccination is an intensity choice $\nu(k)\in[0,qMax]$ with quadratic cost, taken as

$$\text{cost} = -\frac{\gamma}{2}q(k)^2.$$
In the code, `qMax` is a numerical cap on vaccination intensity (default `100.0`).

#### Health-state transitions

The HJB coupling uses the following transition rates (as implemented in the sparse linear system assembly):

- From $S$:
      - $S\to I$ at $q_{SI}(k)=\beta \ell_S(k) LI$
      - $S\to R$ at $q_{SR}(k)=\nu(k)$
- From $I$:
      - $I\to C$ at $\sigma_1$
      - $I\to R$ at $\sigma_3$
      - $I\to S$ at $\mu$
- From $C$:
      - $C\to R$ at $\sigma_2$
      - $C\to S$ at $\alpha_{\text{Epi}}+\mu$
- From $R$:
      - $R\to S$ at $\lambda+\mu$

#### Flow utility (implemented)

For $x\in\{S,I,R\}$:
$$u_x(k)=\theta\log(c_x(k))+(1-\theta)\log(1-\ell_x(k)) - \mathbf{1}_{x=I}d_I,$$
and for susceptibles add the vaccination cost $-(\gamma/2)\nu(k)^2$.

For contained $C$ (contained agents have no labor choice in the code and no leisure term):
$$u_C(k)=\theta\log(c_C(k)) - d_C.$$

### Parameters
Key economic parameters:
- `ρ`: Discount rate
- `θ`: Consumption-leisure preference parameter
- `γ`: Cost of vaccination
- `qMax`: Numerical cap on vaccination intensity (default `100.0`)
- `δ`: Capital depreciation rate
- `ηS, ηI, ηC, ηR`: State-specific productivity levels

Key epidemiological parameters:
- `β`: Infection rate
- `σ1, σ2, σ3`: Transition rates (I→C, C→R, I→R)
- `λ`: Loss of immunity rate
- `μ`: Birth-death rate
- `αEpi`: Exit rate from containment (C→S, interpreted as death + replacement)

Key FP/KFE numerical parameters:
- `T_End`: End time for the FP time-marching
- `Δt`: FP time step (the code uses `Nstep = ceil(T_End/Δt)` and then steps with `Δt_eff = T_End/Nstep`)
- `HJB_every`: Recompute stationary HJB+wage every `HJB_every` FP steps (set to `1` for fully coupled)

## Project Structure

```
├── L_parameters.jl         # Model parameters and structure
├── L_diff.jl               # Finite difference schemes
├── L_HJBsolver.jl          # Value iteration for HJB equations
├── L_wageSolver.jl         # Wage fixed point (uses Roots.jl)
├── L_aggregateVariables.jl # Optimal labor + aggregates (K, L)
├── L_KFEsolver.jl          # Forward Kolmogorov / Fokker--Planck solver (distribution dynamics)
├── L_plots.jl              # Visualization utilities
├── L_loadAll.jl            # Load all modules
├── main.jl                 # Main execution script
├── DEBUG_HJB.jl            # Regression script for stationary HJB+wage solve
├── DEBUG_FP.jl             # Regression script for coupled FP time-marching
├── Project.toml            # Julia dependencies
└── Manifest.toml           # Dependency versions
```

## Installation

1. Clone the repository
2. Open Julia REPL in the project directory
3. Activate the environment:
```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

## Usage

```julia
# Load all modules
include("L_loadAll.jl")

# Initialize parameters
p = MFGEpiEcon(Float64)

# Set up initial value functions
V0 = (VS = zeros(p.Nk), VI = zeros(p.Nk), 
      VC = zeros(p.Nk), VR = zeros(p.Nk))

# Stationary solve given an exogenous distribution Ft
V_solution = value_iterationHJB(V0, Ft, p)

# Dynamic distribution: FP time-marching coupled with stationary HJB
t, Fts, prices = simulate_FP(F0, V0, p)
```

## Numerical Methods

### Finite Differences
- **Derivative approximation**: `∂k_safe` computes a central-difference approximation of $V'(k)$ and floors it away from zero.
- **Advection term**: The capital drift term $b(k) V'(k)$ is discretized with an upwind stencil based on the sign of the drift.

### HJB Solve and Fixed Points
- **Inner solve (given wage)**: For fixed `w`, policies are computed pointwise from the current value function iterate, then the discretized HJB system is solved implicitly as a sparse linear system
      $$(\rho I - A(b) - Q) V = u.$$
- Here $V\in\mathbb{R}^{4N_k}$ is the stacked value vector $V=\big(V_S, V_I, V_C, V_R\big)$ evaluated on the $k$-grid. The objects in the linear system are:
      - **$u$ (flow payoff vector)**: the stacked flow utilities on the grid,
            $$u = \big(u_S, u_I, u_C, u_R\big),$$
            where each component is computed from the current policy rules (consumption, labor, vaccination) as described above.
      - **$A(b)$ (capital drift / advection operator)**: the upwind finite-difference matrix representing the term $b_x(k) \partial_k V_x(k)$ for each health state $x$. It depends on the state-specific drift $b_x$ and uses an upwind stencil chosen by the sign of the drift (backward difference if $b\ge 0$, forward difference if $b<0$), with state constraints enforced at the endpoints.
      - **$Q$ (health-state transition generator)**: the continuous-time generator for transitions across $(S,I,C,R)$ at each grid point. For each $k_i$ it contributes
            $$\sum_{y\neq x} q_{xy}(k_i)\big(V_y(k_i)-V_x(k_i)\big)$$
            to the HJB in state $x$, i.e. off-diagonal entries carry the transition rates $q_{xy}(k_i)$ and diagonal entries carry the total exit rate $\sum_{y\neq x} q_{xy}(k_i)$.
- **Outer wage fixed point**: Wage is updated from the aggregate mapping implied by `Ft` and current policies with damping.
- **Boundary state constraints**: The scheme enforces “no drift leaving the domain” at $k=0$ and $k=k_{\max}$ by consistent endpoint control/drift handling.

### Forward Kolmogorov / Fokker--Planck (FP) Time Marching

Given the optimal controls $c^\*(t,k,x)$, $\ell^\*(t,k,x)$, and $q^\*(t,k)$ implied by the stationary HJB at time $t$ (and current prices $w_t,r_t$). For the FP see the forthcoming paper. 
#### FP discretization

The code advances the distribution with implicit Euler (Backward Euler). For a fixed generator $G^n$ at time step $n$ (controls frozen),

$$\phi^{n+1} = (I - \Delta t G^n)^{-1}\phi^n$$

The drift term is discretized in conservative upwind form (finite-volume style) with a no-flux boundary condition consistent with the HJB state constraints, and health-state transitions are applied as local (in $k$) Markov flows.

### Full Coupled Numerical Scheme (FP + HJB + Wage)

The numerical method is a three-level scheme with nested fixed points:

1. **Time marching (FP)**: for $n=0,1,\dots$ advance $F^n \to F^{n+1}$.
2. **Wage fixed point (at each FP update)**: given $F^n$, solve for the competitive wage $w^n$ by fixed point iteration with damping.
3. **HJB fixed point (given wage)**: for each candidate wage $w$ inside the wage iteration, solve the stationary HJB via damped value iteration where each step solves
      $$(\rho I - A(b) - Q)V = u.$$

Concretely, at time step $n$ the code does:

1. **(Optional) Update stationary policies**, controlled by `HJB_every`.
      Given $F^n$, solve the stationary equilibrium (wage fixed point + stationary HJB) to obtain $V$ and policies.
      Inside the wage fixed point, with wage iterate $w$ and implied wage $w^{\mathrm{imp}}$, update using damping
      
      $$w \leftarrow (1-\omega_w) w + \omega_w w^{\mathrm{imp}}.$$

2. **Build and freeze the forward generator** $G^n$ from $(F^n, V, w)$.

3. **Implicit Euler FP step** (Backward Euler): solve

      $$(I - \Delta t G^n) \phi^{n+1} = \phi^n,$$
      and unstack $\phi^{n+1}$ into $F^{n+1}$.

For speed, the implementation allows recomputing the stationary HJB+wage only every `HJB_every` FP steps (policies frozen in between); setting `HJB_every = 1` yields the fully coupled version.

## Dependencies

- Julia ≥ 1.8
- LinearAlgebra
- Parameters
- Plots
- JuMP
- DifferentialEquations
- OhMyREPL
- ProgressMeter
- BenchmarkTools
- LoopVectorization

## License

[Add your license information here]

## Citation

[Add citation information if applicable]

## Contact

[Add contact information]
