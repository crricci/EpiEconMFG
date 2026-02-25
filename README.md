# EpiEconMFG: Epidemic-Economic Mean Field Game Solver

A Julia implementation of a Mean Field Game (MFG) framework for modeling epidemic-economic dynamics with heterogeneous agents and capital accumulation.

## Overview

This project solves a coupled system of Hamilton-Jacobi-Bellman (HJB) equations and Fokker-Planck equations to analyze the interactions between epidemiological dynamics and economic behavior in a heterogeneous agent framework.

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
- **Value Iteration**: Damped fixed-point iteration for stationary HJB equations
- **Upwind Finite Differences**: Numerical scheme for capital accumulation

## Model Structure

### HJB Equations
The model solves four coupled stationary HJB equations (one for each epidemiological state) with:
- Flow utility from consumption and leisure
- Capital accumulation dynamics
- State transitions (infection, recovery, loss of immunity)
- Piecewise formulation based on labor participation

### Parameters
Key economic parameters:
- `ρ`: Discount rate
- `θ`: Consumption-leisure preference parameter
- `γ`: Cost of vaccination
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
# Note: the forward equation / Fokker–Planck (KFE) side is not implemented yet in this repository.
V_solution = value_iterationHJB(V0, Ft, p)
```

## Numerical Methods

### Finite Differences
- **Log derivatives**: Central differences with positivity enforcement for `log(V'(k))`
- **Flux derivatives**: Upwind scheme based on drift direction for `V'(k) * b(k)`

### Value Iteration
- Damped fixed-point iteration with parameter `ω ∈ (0, 1]`
- Convergence tolerance: `1e-6`
- Maximum iterations: `10,000`

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
