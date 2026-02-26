@with_kw mutable struct MFGEpiEcon{T} 

    # EPI PARAMETERS
    β::T = 0.5            # infection rate
    μ::T = 0.1            # natural birth-death rate
    σ1::T = 0.2           # symptoms development rate (I → C)
    σ2::T = 0.1           # recovery rate from infection (C → R)
    σ3::T = 0.05          # transition rate from I → R
    λ::T = 0.05           # loss of immunity rate (R → S)
    αEpi::T = 0.15        # death rate from contained state (C → S)


    # ECON PARAMETERS
    ρ::T = 0.05           # discount rate
    δ::T = 0.05           # capital depreciation rate
    α::T = 0.6            # Production function 
    A::T = 1.0            # Total factor productivity
    dI::T = 0.1           # disutility of being Infected
    dC::T = 0.2           # disutility of being Contained
    γ::T = 1.0            # coefficient quadratic cost of propensity to vaccination
    qMax::T = 100.0       # cap on vaccination intensity for numerics (q >= 0, bounded above by qMax)
    θ::T = 0.5            # preference consumption vs leisure [0,1]
    ηS::T = 1.0           # productivity of Susceptible agents (benchmark)
    ηI::T = 0.7           # reduced productivity of Infected agents (<1)
    ηC::T = 0.0           # productivity of Contained agents (do not produce)
    ηR::T = 1.0           # productivity of Recovered agents (benchmark)

    # NUMERICAL
    # Capital domain
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
    HJB_every::Int = 5          # recompute stationary HJB+wage every HJB_every FP steps (set to 1 for fully coupled)

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

Returns a `NamedTuple` `(ϕSt, ϕIt, ϕCt, ϕRt)` where each component is a length-`p.Nk`
vector, normalized so that total mass integrates to 1 on the capital grid.
"""
function create_test_distribution(p)
    St = 0.7 * ones(p.Nk)
    It = 0.1 * ones(p.Nk)
    Ct = 0.1 * ones(p.Nk)
    Rt = 0.1 * ones(p.Nk)
    Mass = sum(St + It + Ct + Rt) * p.Δk
    St .= St ./ Mass
    It .= It ./ Mass
    Ct .= Ct ./ Mass
    Rt .= Rt ./ Mass
    return (ϕSt = St, ϕIt = It, ϕCt = Ct, ϕRt = Rt)
end

