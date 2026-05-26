@with_kw mutable struct MFGEpiEcon{T} 

    # EPI PARAMETERS
    # All epidemiological parameters are continuous-time transition rates in units of 1/year.
    # With time measured in years, a mean duration of D days corresponds to a rate 365/D.
    #
    # Model mapping (see src/solvers/fp_kfe.jl):
    # - S -> I at rate infection_rate(k) = β * lS(k) * LI, so when lS≈lI≈1 this behaves like β * S * I.
    # - I exits at rate (σ1 + σ3 + μ): I->C with prob σ1/(σ1+σ3+μ), I->R with prob σ3/(...).
    # - C exits at rate (αEpi + σ2 + μ): C->S at rate (αEpi+μ) (interpretable as death+replacement), C->R at σ2.
    # - R -> S at rate (λ + μ).
    β::T = 200.0        # transmission parameter
    μ::T = 1/70         # background turnover (set ~0 on COVID timescales)
    σ1::T = 365/7       # I → C
    σ2::T = 365/7       # C → R
    σ3::T = 365/14      # I → R
    λ::T = 0.75         # waning immunity R → S
    αEpi::T = 10/70     # additional C → S hazard (tuned so death-probability while in C is small)


    # ECON PARAMETERS
    ρ::T = 0.05           # discount rate
    δ::T = 0.05           # capital depreciation rate
    α::T = 0.35           # Production function
    A::T = 1.0            # Total factor productivity
    dI::T = 0.1           # disutility of being Infected
    dC::T = 0.2           # disutility of being Contained
    γ::T = 10.0           # coefficient quadratic cost of propensity to vaccination
    ξ::T = 0.001          # monetary cost per unit of vaccination intensity ξ(t,k); default constant
    qMax::T = 100.0       # cap on vaccination intensity for numerics (q >= 0, bounded above by qMax)
    θ::T = 0.9            # preference consumption vs leisure [0,1]
    ηS::T = 1.0           # productivity of Susceptible agents (benchmark)
    ηI::T = 0.7           # reduced productivity of Infected agents (<1)
    ηC::T = 0.0           # productivity of Contained agents (do not produce)
    ηR::T = 1.0           # productivity of Recovered agents (benchmark)

    # NUMERICAL
    # Capital domain
    mK::T = 9.0               # mode of the (initial) capital distribution over k (must be > 0)
    σK::T = 0.6               # lognormal dispersion of the initial capital distribution
    N::T = 1e4                # reference population size used to convert initial counts into shares
    I0::T = 1.0               # initial number of infected agents; initial infected share is I0 / N
    MaxK::T = 100.0             # maximum capital level
    Δk::T = 1e-0                # capital step size
    Nk::Int = Int(MaxK/Δk)+1    # number of capital grid points
    k::LinRange{T, Int64} = LinRange(0,MaxK,Nk) # capital grid

    # Temporal domain
    T_End::T = 4.0            # End time (measured in years)
    t_save::LinRange{T, Int64} = LinRange(0,T_End,1000)

    # numerical FP/KFE solver (distribution dynamics)
    Δt::T = 0.05               # time step for FP on [0, T_End] (Nstep is derived)
    FP_Nstep::Int = Int(ceil(T_End / Δt))  # derived default number of FP steps on [0, T_End]
    HJB_every::Int = 1          # recompute stationary HJB+wage every HJB_every FP steps (set to 1 for fully coupled)

    # numerical HJB solver
    ϵDkUp::T = 1e-8          # safe derivative for V'(k)
    ω::T = 1e-1               # damping parameter value iteration HJB (0 < ω ≤ 0.5) 
    tolHJBvalue::T = 1e-6    # convergence tolerance for value iteration HJB
    maxitHJBvalue::Int = Int(1e4)   # maximum number of iterations for value iteration HJB
    w_start::T = 15.0          # initial guess for wage in fixed point iteration

    # outer fixed point (general equilibrium wage)
    ωw::T = 0.2                # damping for wage updates
    tolWage::T = 1e-3          # convergence tolerance for wage fixed point
    maxitWage::Int = 500        # maximum iterations for wage fixed point

    # dynamic forward-backward solver
    maxIterDynamic::Int = 100
    tolDynamic::T = 1e-6
    ωF_dynamic::T = 0.05
    ωV_dynamic::T = 0.25
    maxIterHJBDynamic::Int = 30
    tolHJBDynamic::T = 1e-7
    ωHJBDynamic::T = 0.5
    dynamicInitialGuess::Symbol = :quasistatic
    dynamicTerminal::Symbol = :fixed_quasistatic
    dynamicVerbose::Bool = true


    # progress (when verbose=false but you still want to monitor iteration counters)
    progressWage_every::Int = 5   # show wage iteration counter every this many wage FP iterations
    progressHJB_every::Int = 20   # show HJB value-iteration counter every this many HJB iterations

    # plotting
    truncateKPlots::Bool = true    # plot k-dependent figures only up to an initial-mass quantile
    plotKMassLevel::T = 0.9        # initial population mass shown on k-dependent figures
    
    # general
    verbose::Bool = false

end

"""
    wage(K, L, p)

Compute the competitive wage implied by the Cobb–Douglas production function.

Inputs can be scalars or arrays; `L` is floored at `p.ϵDkUp` for numerical safety.
"""
function wage(K, L, p)
    Ls = max(L, p.ϵDkUp)
    return (1-p.α) * p.A * K.^p.α  .* Ls.^(-p.α)
end

"""
    returns(K, L, p)

Compute the (gross) marginal product of capital implied by the Cobb–Douglas production function.

Inputs can be scalars or arrays; `L` is floored at `p.ϵDkUp` for numerical safety.
"""
function returns(K, L, p)
    Ls = max(L, p.ϵDkUp)
    return p.α * p.A * K.^(p.α-1) .* Ls.^(1-p.α)
end

"""
    vaccine_monetary_cost(t, k, p)

Exogenous monetary cost `ξ(t,k)` per unit of vaccination intensity.

The current calibration is constant and controlled by `p.ξ`; the function form keeps
the call sites ready for a time- and capital-dependent schedule.
"""
function vaccine_monetary_cost(t, k, p)
    return p.ξ
end

"""
    create_test_distribution(p)

Create a simple initial distribution over epidemiological states.

Within each compartment, the distribution over capital is lognormal with mode `p.mK`.
Each compartment mass integrates to the scalar share specified at the top of the function.

Returns a `NamedTuple` `(ϕSt, ϕIt, ϕCt, ϕRt)` where each component is a length-`p.Nk`
vector, normalized so that total mass integrates to 1 on the capital grid.
"""
function create_test_distribution(p)
    # Early-epidemic initial condition (shares; total mass integrates to 1).
    # Keep C and R near zero and start with a small prevalence of I.
    if !(p.N > 0)
        throw(ArgumentError("p.N must be > 0 (got $(p.N))"))
    end
    if !(p.I0 >= 0)
        throw(ArgumentError("p.I0 must be >= 0 (got $(p.I0))"))
    end

    i0 = p.I0 / p.N
    c0 = 0.0
    r0 = 0.0
    s0 = 1.0 - i0 - c0 - r0
    if !(s0 >= 0)
        throw(ArgumentError("Initial susceptible share is negative: I0/N=$(i0). Require I0 <= N."))
    end

    if !(p.mK > 0)
        throw(ArgumentError("p.mK must be > 0 (got $(p.mK))"))
    end
    if !(p.σK > 0)
        throw(ArgumentError("p.σK must be > 0 (got $(p.σK))"))
    end

    # Lognormal density over capital with mode p.mK.
    # For X ~ LogNormal(μ, σ), mode = exp(μ - σ^2) => μ = log(mode) + σ^2.
    μK = log(p.mK) + p.σK^2
    invσ = inv(p.σK)
    invsqrt2π = inv(sqrt(2 * π))

    base = similar(collect(p.k))
    @inbounds for (idx, kval) in pairs(p.k)
        if kval <= 0
            base[idx] = zero(eltype(base))
        else
            z = (log(kval) - μK) * invσ
            base[idx] = invsqrt2π * invσ * exp(-0.5 * z^2) / kval
        end
    end
    base_mass = sum(base) * p.Δk
    if !(base_mass > 0)
        throw(ArgumentError("lognormal base density has zero mass on grid; adjust p.mK/MaxK/Δk"))
    end
    base ./= base_mass  # now integrates to 1 on the k-grid

    St = s0 .* base
    It = i0 .* base
    Ct = c0 .* base
    Rt = r0 .* base

    # Numerical safety: enforce the requested compartment masses (within floating error).
    St .*= (s0 / (sum(St) * p.Δk))
    It .*= (i0 / (sum(It) * p.Δk))
    Ct .*= (c0 == 0 ? 0.0 : (c0 / (sum(Ct) * p.Δk)))
    Rt .*= (r0 == 0 ? 0.0 : (r0 / (sum(Rt) * p.Δk)))
    return (ϕSt = St, ϕIt = It, ϕCt = Ct, ϕRt = Rt)
end
