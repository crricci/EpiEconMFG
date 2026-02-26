# EpiEconMFG: Epidemic-Economic Mean Field Game Solver

A Julia implementation of a Mean Field Game (MFG) framework for modeling epidemic-economic dynamics with heterogeneous agents and capital accumulation.

## Overview

This project solves a system of stationary Hamilton-Jacobi-Bellman (HJB) equations on a capital grid, coupled through an exogenously provided cross-sectional distribution `Ft` (used for aggregates and the infection externality) and an outer fixed point for the competitive wage.

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

### Solved Stationary System (Implemented)

Let $x \in \{S,I,C,R\}$ denote the epidemiological state and $k \in [0, k_{\max}]$ the individual capital state (discretized on a grid).

The code solves the stationary HJB system
$$
\rho V_x(k) 
= u_x(k)
+ b_x(k)\,\partial_k V_x(k)
+ \sum_{y\neq x} q_{xy}(k)\,\big(V_y(k)-V_x(k)\big),
\qquad x\in\{S,I,C,R\},
$$
with capital drift
$$
b_x(k) = (r-\delta)k + \eta_x\,w\,\ell_x(k) - c_x(k) \quad (x\in\{S,I,R\}),
\qquad b_C(k) = (r-\delta)k - c_C(k).
$$

`Ft = (ϕSt, ϕIt, ϕCt, ϕRt)` is treated as an input distribution over states and the $k$-grid and is used to compute aggregates and the infection externality.

#### Controls

Consumption is taken from the FOC
$$
c_x(k) = \frac{\theta}{\partial_k V_x(k)}.
$$

Labor is chosen (and then clamped to $[0,1]$) by the closed-form rule
$$
\ell_x(k) = \Pi_{[0,1]}\left(1 - \frac{1-\theta}{\partial_k V_x(k)\,W_x(k)}\right),
$$
where effective wages are
$$
W_I = \eta_I w,\quad W_R = \eta_R w,\quad W_C = 0,
$$
and for susceptibles (capturing the infection externality wedge)
$$
W_S(k) = \eta_S w + \beta\,LI\,\frac{V_I(k)-V_S(k)}{\partial_k V_S(k)},
\qquad LI = \int \ell_I(k)\,\phi_I(k)\,dk.
$$

Vaccination is an intensity choice $\nu(k)\in[0,qMax]$ with quadratic cost, taken as
$$
\nu(k) = \Pi_{[0,qMax]}\left(\frac{V_R(k)-V_S(k)}{\gamma}\right),
\qquad \text{cost} = -\frac{\gamma}{2}\nu(k)^2.
$$
In the code, `qMax` is a numerical cap on vaccination intensity (default `100.0`).

#### Health-state transitions

The HJB coupling uses the following transition rates (as implemented in the sparse linear system assembly):

- From $S$:
      - $S\to I$ at $q_{SI}(k)=\beta\,\ell_S(k)\,LI$
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
$$
u_x(k)=\theta\log(c_x(k))+(1-\theta)\log(1-\ell_x(k)) - \mathbf{1}_{x=I}d_I,
$$
and for susceptibles add the vaccination cost $-(\gamma/2)\nu(k)^2$.

For contained $C$ (contained agents have no labor choice in the code and no leisure term):
$$
u_C(k)=\theta\log(c_C(k)) - d_C.
$$

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

## Project Structure

```
├── L_parameters.jl         # Model parameters and structure
├── L_diff.jl               # Finite difference schemes
├── L_HJBsolver.jl          # Value iteration for HJB equations
├── L_wageSolver.jl         # Wage fixed point (uses Roots.jl)
├── L_aggregateVariables.jl # Optimal labor + aggregates (K, L)
├── L_plots.jl              # Visualization utilities
├── L_loadAll.jl            # Load all modules
├── main.jl                 # Main execution script
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

# Solve HJB equations (requires an exogenous distribution Ft)
V_solution = value_iterationHJB(V0, Ft, p)
```

## Numerical Methods

### Finite Differences
- **Derivative approximation**: `∂k_log` computes a central-difference approximation of $V'(k)$ and floors it away from zero.
- **Advection term**: The capital drift term $b(k)\,V'(k)$ is discretized with an upwind stencil based on the sign of the drift.

### HJB Solve and Fixed Points
- **Inner solve (given wage)**: For fixed `w`, policies are computed pointwise from the current value function iterate, then the discretized HJB system is solved implicitly as a sparse linear system
      $$ (\rho I - A(b) - Q) V = u. $$
- Here $V\in\mathbb{R}^{4N_k}$ is the stacked value vector $V=\big(V_S, V_I, V_C, V_R\big)$ evaluated on the $k$-grid. The objects in the linear system are:
      - **$u$ (flow payoff vector)**: the stacked flow utilities on the grid,
            $$u = \big(u_S, u_I, u_C, u_R\big),$$
            where each component is computed from the current policy rules (consumption, labor, vaccination) as described above.
      - **$A(b)$ (capital drift / advection operator)**: the upwind finite-difference matrix representing the term $b_x(k)\,\partial_k V_x(k)$ for each health state $x$. It depends on the state-specific drift $b_x$ and uses an upwind stencil chosen by the sign of the drift (backward difference if $b\ge 0$, forward difference if $b<0$), with state constraints enforced at the endpoints.
      - **$Q$ (health-state transition generator)**: the continuous-time generator for transitions across $(S,I,C,R)$ at each grid point. For each $k_i$ it contributes
            $$\sum_{y\neq x} q_{xy}(k_i)\big(V_y(k_i)-V_x(k_i)\big)$$
            to the HJB in state $x$, i.e. off-diagonal entries carry the transition rates $q_{xy}(k_i)$ and diagonal entries carry the total exit rate $\sum_{y\neq x} q_{xy}(k_i)$.
- **Outer wage fixed point**: Wage is updated from the aggregate mapping implied by `Ft` and current policies with damping.
- **Boundary state constraints**: The scheme enforces “no drift leaving the domain” at $k=0$ and $k=k_{\max}$ by consistent endpoint control/drift handling.

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
