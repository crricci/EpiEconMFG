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
    dI::T = 15.0          # disutility of being Infected
    dC::T = 30.0          # disutility of being Contained
    γ::T = 10.0           # coefficient quadratic cost of propensity to vaccination
    ξ::T = 0.25           # monetary cost per unit of vaccination intensity ξ(t,k); default constant
    vaccineCostProfile::Symbol = :constant # :constant or :linear_k
    ξKMin::T = 0.0        # lower endpoint for :linear_k vaccine cost at k=0
    ξKMax::T = 0.25       # upper endpoint for :linear_k vaccine cost at k=MaxK
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
    useCsvInitialDistribution::Bool = true # if true, load the initial distribution from distributions.csv
    initialDistributionCsvDir::String = "data/initial_conditions/dynamic_NoEpidemic_long_restart_from_finalS"
    MaxK::T = 20.0             # maximum capital level
    Δk::T = 2*1e-1                # capital step size
    Nk::Int = Int(MaxK/Δk)+1    # number of capital grid points
    k::LinRange{T, Int64} = LinRange(0,MaxK,Nk) # capital grid

    # Temporal domain
    T_End::T = 5.0            # End time (measured in years)
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
    maxitWage::Int = 5000        # maximum iterations for wage fixed point

    # dynamic forward-backward solver
    maxIterDynamic::Int = 10000
    tolDynamic::T = 1e-6
    ωF_dynamic::T = 0.05
    ωV_dynamic::T = 0.25
    maxIterHJBDynamic::Int = 1000
    tolHJBDynamic::T = 1e-7
    ωHJBDynamic::T = 0.5
    dynamicInitialGuess::Symbol = :quasistatic
    dynamicTerminal::Symbol = :fixed_quasistatic
    dynamicVerbose::Bool = true


    # progress (when verbose=false but you still want to monitor iteration counters)
    progressWage_every::Int = 5   # show wage iteration counter every this many wage FP iterations
    progressHJB_every::Int = 20   # show HJB value-iteration counter every this many HJB iterations

    # plotting
    truncateKPlots::Bool = true    # plot k-dependent figures on an initial-mass interval
    cutKMassLevelBottom::T = 0.025  # lower initial population mass cut for k-dependent figures
    cutKMassLevelTop::T = 1.0     # upper initial population mass cut for k-dependent figures
    plotVaccinationLogScale::Bool = true # plot vaccination intensity as log10(1+q)
    
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

When `p.vaccineCostProfile == :constant`, the cost is controlled by `p.ξ`.
When `p.vaccineCostProfile == :linear_k`, the cost is linear in capital from
`p.ξKMin` at `k=0` to `p.ξKMax` at `k=p.MaxK`.
"""
function vaccine_monetary_cost(t, k, p)
    if p.vaccineCostProfile == :constant
        return p.ξ
    elseif p.vaccineCostProfile == :linear_k
        if !(p.MaxK > 0)
            throw(ArgumentError("p.MaxK must be > 0 for vaccineCostProfile=:linear_k"))
        end
        kshare = clamp(k / p.MaxK, 0.0, 1.0)
        return p.ξKMin + (p.ξKMax - p.ξKMin) * kshare
    end

    throw(ArgumentError("Unsupported vaccineCostProfile=$(p.vaccineCostProfile). Use :constant or :linear_k."))
end

function _project_root()
    return normpath(joinpath(@__DIR__, "..", ".."))
end

function _resolve_project_path(path::AbstractString)
    return isabspath(path) ? normpath(path) : normpath(joinpath(_project_root(), path))
end

function _parse_csv_header(line::AbstractString)
    return split(chomp(line), ",")
end

function _csv_column(header, name::AbstractString, path::AbstractString)
    idx = findfirst(==(name), header)
    idx === nothing && error("Missing CSV column '$name' in $path")
    return idx
end

function _initial_compartment_shares(p)
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
    return (s0 = s0, i0 = i0, c0 = c0, r0 = r0)
end

function _split_wealth_distribution(base, p)
    shares = _initial_compartment_shares(p)
    return (
        ϕSt = shares.s0 .* base,
        ϕIt = shares.i0 .* base,
        ϕCt = shares.c0 .* base,
        ϕRt = shares.r0 .* base,
    )
end

"""
    load_initial_distribution_csv(dir, p)

Load a wealth distribution from `dir/distributions.csv` and use it as the model
initial condition.

The loader takes the last time block in the file, validates it against `p.k`,
collapses the CSV compartments into the marginal wealth density
`phiS + phiI + phiC + phiR`, and normalizes that density to mass one. The
epidemiological split is then generated from the current parameters: the initial
infected mass is `p.I0 / p.N`, the contained/recovered masses are zero, and the
remaining mass is susceptible.
"""
function load_initial_distribution_csv(dir::AbstractString, p)
    source_dir = _resolve_project_path(dir)
    path = joinpath(source_dir, "distributions.csv")
    if !isfile(path)
        error("Missing initial-condition CSV: $path")
    end

    S = zeros(Float64, p.Nk)
    I = zeros(Float64, p.Nk)
    C = zeros(Float64, p.Nk)
    R = zeros(Float64, p.Nk)
    current_time_index = nothing
    current_time = NaN
    rows_in_block = 0

    open(path, "r") do io
        header_line = readline(io)
        header = _parse_csv_header(header_line)
        it = _csv_column(header, "time_index", path)
        tt = _csv_column(header, "t", path)
        ik = _csv_column(header, "k_index", path)
        kk = _csv_column(header, "k", path)
        iS = _csv_column(header, "phiS", path)
        iI = _csv_column(header, "phiI", path)
        iC = _csv_column(header, "phiC", path)
        iR = _csv_column(header, "phiR", path)

        for (line_no, line) in enumerate(eachline(io))
            isempty(strip(line)) && continue
            row = split(line, ",")
            length(row) >= length(header) || error("Malformed CSV line $(line_no + 1) in $path")

            time_index = parse(Int, strip(row[it]))
            if current_time_index === nothing || time_index != current_time_index
                fill!(S, 0.0)
                fill!(I, 0.0)
                fill!(C, 0.0)
                fill!(R, 0.0)
                current_time_index = time_index
                current_time = parse(Float64, strip(row[tt]))
                rows_in_block = 0
            end

            k_index = parse(Int, strip(row[ik]))
            if !(1 <= k_index <= p.Nk)
                error("k_index=$k_index outside 1:$(p.Nk) in $path line $(line_no + 1). Check MaxK/Δk.")
            end

            kval = parse(Float64, strip(row[kk]))
            expected_k = Float64(p.k[k_index])
            if !isapprox(kval, expected_k; atol = max(1e-10, 1e-8 * max(abs(kval), abs(expected_k))), rtol = 1e-8)
                error(
                    "Grid mismatch in $path line $(line_no + 1): " *
                    "CSV k=$kval but p.k[$k_index]=$expected_k. Check MaxK/Δk."
                )
            end

            S[k_index] = max(parse(Float64, strip(row[iS])), 0.0)
            I[k_index] = max(parse(Float64, strip(row[iI])), 0.0)
            C[k_index] = max(parse(Float64, strip(row[iC])), 0.0)
            R[k_index] = max(parse(Float64, strip(row[iR])), 0.0)
            rows_in_block += 1
        end
    end

    if current_time_index === nothing || rows_in_block == 0
        error("No distribution rows found in $path")
    end
    if rows_in_block != p.Nk
        error("Last CSV time block in $path has $rows_in_block rows; expected p.Nk=$(p.Nk)")
    end

    base = S .+ I .+ C .+ R
    total_mass = sum(base) * p.Δk
    if !(isfinite(total_mass) && total_mass > 0)
        error("Non-positive total mass in CSV initial condition $path at time_index=$current_time_index")
    end

    base ./= total_mass

    if p.verbose
        shares = _initial_compartment_shares(p)
        println(
            "Loaded CSV initial distribution from $source_dir " *
            "(time_index=$current_time_index, t=$current_time, wealth_mass=$total_mass, " *
            "S0=$(shares.s0), I0_share=$(shares.i0))"
        )
    end

    return _split_wealth_distribution(base, p)
end

"""
    initial_distribution(p)

Return the model initial distribution. By default this is the generated
lognormal distribution. If `p.useCsvInitialDistribution` is true, it is loaded
from `p.initialDistributionCsvDir`.
"""
function initial_distribution(p)
    if p.useCsvInitialDistribution
        return load_initial_distribution_csv(p.initialDistributionCsvDir, p)
    end
    return create_test_distribution(p)
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
    shares = _initial_compartment_shares(p)

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

    F0 = _split_wealth_distribution(base, p)

    # Numerical safety: enforce the requested compartment masses (within floating error).
    # Zero-mass compartments must stay exactly zero; otherwise 0/0 would create NaNs.
    function normalize_compartment!(density, target_mass, name)
        if target_mass == 0
            fill!(density, 0.0)
            return density
        end

        current_mass = sum(density) * p.Δk
        if !(isfinite(current_mass) && current_mass > 0)
            throw(ArgumentError("Cannot normalize initial $name mass: current_mass=$current_mass target_mass=$target_mass"))
        end
        density .*= target_mass / current_mass
        return density
    end

    normalize_compartment!(F0.ϕSt, shares.s0, "S")
    normalize_compartment!(F0.ϕIt, shares.i0, "I")
    normalize_compartment!(F0.ϕCt, shares.c0, "C")
    normalize_compartment!(F0.ϕRt, shares.r0, "R")
    return F0
end
