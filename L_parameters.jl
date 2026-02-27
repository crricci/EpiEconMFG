@with_kw mutable struct MFGEpiEcon{T} 

    # EPI PARAMETERS
    # All epidemiological parameters are continuous-time transition rates in units of 1/year.
    # With time measured in years, a mean duration of D days corresponds to a rate 365/D.
    #
    # Model mapping (see L_KFEsolver.jl):
    # - S -> I at rate infection_rate(k) = β * lS(k) * LI, so when lS≈lI≈1 this behaves like β * S * I.
    # - I exits at rate (σ1 + σ3 + μ): I->C with prob σ1/(σ1+σ3+μ), I->R with prob σ3/(...).
    # - C exits at rate (αEpi + σ2 + μ): C->S at rate (αEpi+μ) (interpretable as death+replacement), C->R at σ2.
    # - R -> S at rate (λ + μ).
    β::T = 20.0          # transmission parameter (≈ R0*(σ1+σ3) with R0≈3 and mean infectious duration ≈5 days)
    μ::T = 0.0            # background turnover (set ~0 on COVID timescales)
    σ1::T = 0.1          # I → C (mean time ~5 days, with ~60% going to C)
    # σ1::T = 44.0          # I → C (mean time ~5 days, with ~60% going to C)
    σ2::T = 0.4          # C → R (mean time in C ~10 days)
    # σ2::T = 36.5          # C → R (mean time in C ~10 days)
    σ3::T = 0.3          # I → R (mean time ~5 days, with ~40% going directly to R)
    # σ3::T = 29.0          # I → R (mean time ~5 days, with ~40% going directly to R)
    λ::T = 0.33           # waning immunity R → S (mean ~3 years)
    αEpi::T = 0.18        # additional C → S hazard (tuned so death-probability while in C is small)


    # ECON PARAMETERS
    ρ::T = 0.05           # discount rate
    δ::T = 0.05           # capital depreciation rate
    α::T = 0.6            # Production function 
    A::T = 1.0            # Total factor productivity
    dI::T = 0.1           # disutility of being Infected
    dC::T = 0.2           # disutility of being Contained
    γ::T = 10.0           # coefficient quadratic cost of propensity to vaccination
    qMax::T = 100.0       # cap on vaccination intensity for numerics (q >= 0, bounded above by qMax)
    θ::T = 0.75            # preference consumption vs leisure [0,1]
    ηS::T = 1.0           # productivity of Susceptible agents (benchmark)
    ηI::T = 0.7           # reduced productivity of Infected agents (<1)
    ηC::T = 0.0           # productivity of Contained agents (do not produce)
    ηR::T = 1.0           # productivity of Recovered agents (benchmark)

    # NUMERICAL
    # Capital domain
    mK::T = 9.0               # mode of the (initial) capital distribution over k (must be > 0)
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


    # progress (when verbose=false but you still want to monitor iteration counters)
    progressWage_every::Int = 5   # show wage iteration counter every this many wage FP iterations
    progressHJB_every::Int = 20   # show HJB value-iteration counter every this many HJB iterations
    
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
    i0 = 1e-1
    c0 = 0.0
    r0 = 0.0
    s0 = 1.0 - i0 - c0 - r0

    if !(p.mK > 0)
        throw(ArgumentError("p.mK must be > 0 (got $(p.mK))"))
    end

    # Lognormal density over capital with mode p.mK.
    # For X ~ LogNormal(μ, σ), mode = exp(μ - σ^2) => μ = log(mode) + σ^2.
    σK = 0.6
    μK = log(p.mK) + σK^2
    invσ = inv(σK)
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

