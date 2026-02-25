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
    θ::T = 0.5            # preference consumption vs leisure [0,1]
    ηS::T = 1.0           # productivity of Susceptible agents (benchmark)
    ηI::T = 0.7           # reduced productivity of Infected agents (<1)
    ηC::T = 0.0           # productivity of Contained agents (do not produce)
    ηR::T = 1.0           # productivity of Recovered agents (benchmark)

    # NUMERICAL
    # Capital domain
    MaxK::T = 100.0             # maximum capital level
    Δk::T = 1e-1                # capital step size
    Nk::Int = Int(MaxK/Δk)+1    # number of capital grid points
    k::LinRange{T, Int64} = LinRange(0,MaxK,Nk) # capital grid

    # Temporal domain
    T_End::T = 5.00            # End time (measure in years)
    t_save::LinRange{T, Int64} = LinRange(0,T_End,1000)

    # numerical HJB solver
    ϵDkUp::T = 1e-8          # safe derivative for V'(k)
    ω::T = 1e-1               # damping parameter value iteration HJB (0 < ω ≤ 0.5) 
    tolHJBvalue::T = 1e-6    # convergence tolerance for value iteration HJB
    maxitHJBvalue::Int = Int(1e4)   # maximum number of iterations for value iteration HJB
    w_start::T = 15.0          # initial guess for wage in fixed point iteration

    # outer fixed point (general equilibrium wage)
    ωw::T = 5e-2               # damping for wage updates
    tolWage::T = 1e-6          # convergence tolerance for wage fixed point
    maxitWage::Int = 50        # maximum iterations for wage fixed point
    
    
    # general
    verbose::Bool = true

end

function wage(K, L, p)
    Ls = max(L, p.ϵDkUp)
    return (1-p.α) * p.A * K.^p.α  .* Ls.^(-p.α)
end

function returns(K, L, p)
    Ls = max(L, p.ϵDkUp)
    return p.α * p.A * K.^(p.α-1) .* Ls.^(1-p.α)
end
