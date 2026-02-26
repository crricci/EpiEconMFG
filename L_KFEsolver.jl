# Forward Kolmogorov / Fokker-Planck solver (distribution dynamics)
# Coupled to the stationary HJB via optimal controls (c*, l*, q*).

# Stack/unstack helpers for distributions on the k-grid.
"""
    stack_distribution(Ft)

Stack a distribution `NamedTuple` `(ϕSt, ϕIt, ϕCt, ϕRt)` into a single vector of
length `4*p.Nk` (with blocks ordered S, I, C, R).
"""
function stack_distribution(Ft)
    return vcat(Ft.ϕSt, Ft.ϕIt, Ft.ϕCt, Ft.ϕRt)
end

"""
    unstack_distribution(phi, p)

Inverse of `stack_distribution`.

Takes a stacked vector `phi` of length `4*p.Nk` and returns the corresponding
`NamedTuple` `(ϕSt, ϕIt, ϕCt, ϕRt)`.
"""
function unstack_distribution(phi, p)
    Nk = p.Nk
    return (
        ϕSt = phi[1:Nk],
        ϕIt = phi[Nk + 1:2Nk],
        ϕCt = phi[2Nk + 1:3Nk],
        ϕRt = phi[3Nk + 1:4Nk],
    )
end

# Compute policies needed by the forward equation, consistent with the stationary HJB implementation.
"""
    compute_FP_policies(V, Ft, p; w, deriv_cache=nothing)

Compute the policy functions and auxiliary objects needed to build the FP/KFE generator.

This mirrors the stationary HJB control logic (consumption from the FOC, labor from the
static problem, vaccination from the HJB value difference), and enforces state constraints
at the capital-grid boundaries.

# Keyword Arguments
- `w`: wage used in the static labor problem.
- `deriv_cache`: optional `DerivDkCache` to reuse allocations when computing derivatives with respect to capital.

# Returns
A `NamedTuple` containing consumption, labor, drifts, transition intensities, and aggregates.
"""
function compute_FP_policies(V, Ft, p; w, deriv_cache=nothing)

    # Derivatives V'(k) with safe floor (same as HJB)
    ∂V = isnothing(deriv_cache) ? compute_∂V_dk(V, p) : compute_∂V_dk!(deriv_cache, V, p)

    # Labor, aggregates, and prices
    agg = compute_labor_and_aggregates(V, ∂V, Ft, p; w=w)
    lOpt = agg.lOpt
    K = agg.K
    L = agg.L
    r = agg.r
    LI = agg.LI

    capital_income = (r - p.δ) .* p.k

    # Consumption from FOC: θ/c = V'(k) => c = θ / V'(k)
    cS = p.θ ./ ∂V.∂kVS
    cI = p.θ ./ ∂V.∂kVI
    cC = p.θ ./ ∂V.∂kVC
    cR = p.θ ./ ∂V.∂kVR

    # State constraints at the boundaries enforced on controls
    incomeS = capital_income .+ (p.ηS * w) .* lOpt.lS
    incomeI = capital_income .+ (p.ηI * w) .* lOpt.lI
    incomeC = capital_income
    incomeR = capital_income .+ (p.ηR * w) .* lOpt.lR

    cS[1] = min(cS[1], max(incomeS[1], 0.0))
    cI[1] = min(cI[1], max(incomeI[1], 0.0))
    cC[1] = min(cC[1], max(incomeC[1], 0.0))
    cR[1] = min(cR[1], max(incomeR[1], 0.0))

    cS[end] = max(cS[end], max(incomeS[end], 0.0))
    cI[end] = max(cI[end], max(incomeI[end], 0.0))
    cC[end] = max(cC[end], max(incomeC[end], 0.0))
    cR[end] = max(cR[end], max(incomeR[end], 0.0))

    # Drift b(k) = income - consumption
    bS = incomeS .- cS
    bI = incomeI .- cI
    bC = incomeC .- cC
    bR = incomeR .- cR

    # State constraints: no drift leaving domain
    bS[1] = max(bS[1], 0.0); bS[end] = min(bS[end], 0.0)
    bI[1] = max(bI[1], 0.0); bI[end] = min(bI[end], 0.0)
    bC[1] = max(bC[1], 0.0); bC[end] = min(bC[end], 0.0)
    bR[1] = max(bR[1], 0.0); bR[end] = min(bR[end], 0.0)

    # Mean-field infection intensity for S->I at each k
    infection_rate = p.β .* lOpt.lS .* LI

    # Vaccination intensity S->R (same rule as HJB)
    q_rate = clamp.((V.VR .- V.VS) ./ p.γ, 0.0, p.qMax)

    return (
        lOpt = lOpt,
        cS = cS,
        cI = cI,
        cC = cC,
        cR = cR,
        bS = bS,
        bI = bI,
        bC = bC,
        bR = bR,
        infection_rate = infection_rate,
        q_rate = q_rate,
        K = K,
        L = L,
        w = w,
        r = r,
        LI = LI,
    )
end

# Assemble drift operator for forward equation in conservative upwind form.
"""
    add_forward_drift_entries!(I, J, X, offset, b, p)

Append triplet entries for the conservative upwind discretization of the drift term
in the FP/KFE equation.

This builds the sparse operator corresponding to the drift term in conservative form on the capital grid,
with upwind flux splitting based on the sign of `b`.
"""
function add_forward_drift_entries!(I, J, X, offset, b, p)
    Nk = p.Nk
    Δk = p.Δk

    @inbounds begin
        bplus_prev = 0.0
        bminus_prev = 0.0

        for i in 1:Nk
            b_i = b[i]
            bplus_i = max(b_i, 0.0)
            bminus_i = min(b_i, 0.0)

            row = offset + i

            diag = -(bplus_i) / Δk + (i > 1 ? bminus_prev / Δk : 0.0)
            push!(I, row); push!(J, row); push!(X, diag)

            if i > 1
                # inflow from left interface uses bplus at i-1 and phi_{i-1}
                push!(I, row); push!(J, offset + (i - 1)); push!(X, bplus_prev / Δk)
            end

            if i < Nk
                # inflow from right interface uses bminus at i and phi_{i+1}
                push!(I, row); push!(J, offset + (i + 1)); push!(X, -(bminus_i) / Δk)
            end

            bplus_prev = bplus_i
            bminus_prev = bminus_i
        end
    end

    return nothing
end

# Build the full forward generator G for phi_dot = G * phi.
"""
    build_FP_generator(policies, p)

Build the sparse generator `G` for the stacked distribution dynamics:

phi_dot = G * phi.

Includes the conservative upwind drift blocks and the local epidemiological transition
rates at each capital grid point.
"""
function build_FP_generator(policies, p)
    Nk = p.Nk
    n = 4 * Nk

    I = Int[]
    J = Int[]
    X = eltype(policies.bS)[]

    sizehint!(I, 24n)
    sizehint!(J, 24n)
    sizehint!(X, 24n)

    # Drift blocks (S,I,C,R)
    add_forward_drift_entries!(I, J, X, 0 * Nk, policies.bS, p)
    add_forward_drift_entries!(I, J, X, 1 * Nk, policies.bI, p)
    add_forward_drift_entries!(I, J, X, 2 * Nk, policies.bC, p)
    add_forward_drift_entries!(I, J, X, 3 * Nk, policies.bR, p)

    # Local transitions at each k
    @inbounds for i in 1:Nk
        idxS = 0 * Nk + i
        idxI = 1 * Nk + i
        idxC = 2 * Nk + i
        idxR = 3 * Nk + i

        inf = policies.infection_rate[i]
        q = policies.q_rate[i]

        # S equation:
        # dφS += μ φI + (α+μ) φC + (λ+μ) φR - (q + inf) φS
        push!(I, idxS); push!(J, idxS); push!(X, -(q + inf))
        push!(I, idxS); push!(J, idxI); push!(X, p.μ)
        push!(I, idxS); push!(J, idxC); push!(X, p.αEpi + p.μ)
        push!(I, idxS); push!(J, idxR); push!(X, p.λ + p.μ)

        # I equation:
        # dφI += inf φS - (σ1 + σ3 + μ) φI
        push!(I, idxI); push!(J, idxS); push!(X, inf)
        push!(I, idxI); push!(J, idxI); push!(X, -(p.σ1 + p.σ3 + p.μ))

        # C equation:
        # dφC += σ1 φI - (α + σ2 + μ) φC
        push!(I, idxC); push!(J, idxI); push!(X, p.σ1)
        push!(I, idxC); push!(J, idxC); push!(X, -(p.αEpi + p.σ2 + p.μ))

        # R equation:
        # dφR += q φS + σ3 φI + σ2 φC - (λ + μ) φR
        push!(I, idxR); push!(J, idxS); push!(X, q)
        push!(I, idxR); push!(J, idxI); push!(X, p.σ3)
        push!(I, idxR); push!(J, idxC); push!(X, p.σ2)
        push!(I, idxR); push!(J, idxR); push!(X, -(p.λ + p.μ))
    end

    G = sparse(I, J, X, n, n)
    return G
end

"""
    renormalize_distribution!(phi, p)

Renormalize the stacked distribution vector so its total mass integrates to 1.

Returns the pre-normalization mass.
"""
function renormalize_distribution!(phi, p)
    Nk = p.Nk
    mass = (sum(phi[1:Nk]) + sum(phi[Nk + 1:2Nk]) + sum(phi[2Nk + 1:3Nk]) + sum(phi[3Nk + 1:4Nk])) * p.Δk
    if !(isfinite(mass) && mass > 0)
        error("Non-finite or non-positive mass in FP: mass=$mass")
    end
    phi ./= mass
    return mass
end

"""
    project_nonnegative!(phi)

Project the distribution vector onto the nonnegative orthant (elementwise max with 0).
"""
function project_nonnegative!(phi)
    @inbounds for i in eachindex(phi)
        phi[i] = max(phi[i], 0.0)
    end
    return phi
end

"""
Simulate distribution dynamics (FP/KFE) on [0, T_End] using implicit Euler.

- The stationary HJB+wage is re-solved every `HJB_every` steps using the current Ft.
- Between HJB updates, controls and the generator are frozen.

Time discretization:
- Preferred: pass `Δt` (or set `p.Δt`), then `Nstep = ceil(T_End/Δt)` and the actual step used is `T_End/Nstep`.
- Optional override: pass `Nstep` explicitly.

Returns (t, Fts, prices), where
- t::Vector
- Fts::Vector{NamedTuple} (ϕSt,ϕIt,ϕCt,ϕRt)
- prices::NamedTuple (w, r, K, L, LI)
"""
function simulate_FP(F0, V0, p; T_End=p.T_End, Δt=p.Δt, Nstep=nothing, HJB_every=p.HJB_every)
    if HJB_every < 1
        error("HJB_every must be >= 1")
    end

    Nstep_eff = if isnothing(Nstep)
        if !(isfinite(Δt) && Δt > 0)
            error("Δt must be finite and > 0")
        end
        Int(ceil(T_End / Δt))
    else
        Int(Nstep)
    end

    if Nstep_eff < 1
        error("Nstep must be >= 1")
    end

    t = collect(range(0, T_End, length=Nstep_eff + 1))
    Δt_eff = T_End / Nstep_eff

    phi = copy(stack_distribution(F0))
    renormalize_distribution!(phi, p)

    Fts = Vector{Any}(undef, Nstep_eff + 1)
    Fts[1] = unstack_distribution(phi, p)

    ws = fill(NaN, Nstep_eff + 1)
    rs = fill(NaN, Nstep_eff + 1)
    Ks = fill(NaN, Nstep_eff + 1)
    Ls = fill(NaN, Nstep_eff + 1)
    LIs = fill(NaN, Nstep_eff + 1)

    Vguess = (VS = copy(V0.VS), VI = copy(V0.VI), VC = copy(V0.VC), VR = copy(V0.VR))

    dcache = DerivDkCache(eltype(Vguess.VS), p.Nk)

    G = nothing
    Afact = nothing
    last_policies = nothing

    p.verbose && println("FP: T_End=$T_End, requested Δt=$Δt, using Nstep=$Nstep_eff and Δt_eff=$Δt_eff")

    for n in 1:Nstep_eff
        need_update = (n == 1) || ((n - 1) % HJB_every == 0)

        if need_update
            Ft = unstack_distribution(phi, p)

            # Solve stationary HJB + wage fixed point given current distribution
            Vsol = value_iterationHJB(Vguess, Ft, p)
            w = Vsol.w

            # Compute policies and build forward generator
            pol = compute_FP_policies((VS=Vsol.VS, VI=Vsol.VI, VC=Vsol.VC, VR=Vsol.VR), Ft, p; w=w, deriv_cache=dcache)
            G = build_FP_generator(pol, p)

            # Build and factorize implicit Euler matrix once per batch
            nstate = 4 * p.Nk
            A = spdiagm(0 => ones(eltype(phi), nstate)) - Δt_eff * G
            Afact = lu(A)

            last_policies = pol
            Vguess = (VS = Vsol.VS, VI = Vsol.VI, VC = Vsol.VC, VR = Vsol.VR)

            p.verbose && println("FP: updated policies at step $(n-1), t=$(t[n]): w=$(pol.w), r=$(pol.r), LI=$(pol.LI)")
        end

        # implicit Euler step
        phi = Afact \ phi

        # small numerical fixups
        project_nonnegative!(phi)
        renormalize_distribution!(phi, p)

        Fts[n + 1] = unstack_distribution(phi, p)

        if !isnothing(last_policies)
            ws[n + 1] = last_policies.w
            rs[n + 1] = last_policies.r
            Ks[n + 1] = last_policies.K
            Ls[n + 1] = last_policies.L
            LIs[n + 1] = last_policies.LI
        end

        if p.verbose && (n == 1 || n % 25 == 0)
            mass = (sum(phi[1:p.Nk]) + sum(phi[p.Nk+1:2p.Nk]) + sum(phi[2p.Nk+1:3p.Nk]) + sum(phi[3p.Nk+1:4p.Nk])) * p.Δk
            println("FP: step=$n t=$(t[n+1]) mass=$mass min(phi)=$(minimum(phi))")
        end
    end

    prices = (w=ws, r=rs, K=Ks, L=Ls, LI=LIs)
    return t, Fts, prices
end
