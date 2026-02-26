
function value_iterationHJB_given_wage(V0, Ft, p; w)
    """
    V0: NamedTuple
        Initial guess (VS, VI, VC, VR), each Vector{Float64}
    Ft: NamedTuple
        Exogenous distribution over (S,I,C,R) across the k-grid:
        (ϕSt, ϕIt, ϕCt, ϕRt), each a Vector{Float64} of length Nk.
        Used to compute aggregates and infection externality terms.

    p: model parameters (MFGEpiEcon)
    """
    # current iterate
    VS = copy(V0.VS)
    VI = copy(V0.VI)
    VC = copy(V0.VC)
    VR = copy(V0.VR)

    w_fixed = w

    dcache = DerivDkCache(eltype(VS), p.Nk)
    acache = HJBAssemblyCache(eltype(VS), p.Nk)

    for it in 1:p.maxitHJBvalue

        # HJB operator with fixed wage
        Vnew, _ = T_HJB((VS = VS, VI = VI, VC = VC, VR = VR), Ft, p; w0 = w_fixed, deriv_cache = dcache, assembly_cache = acache)

        if !(all(isfinite, Vnew.VS) && all(isfinite, Vnew.VI) && all(isfinite, Vnew.VC) && all(isfinite, Vnew.VR))
            error("Non-finite values in HJB operator at iter=$it (w=$w_fixed).")
        end

        # damped update
        VS_new = (1-p.ω) .* VS .+ p.ω .* Vnew.VS
        VI_new = (1-p.ω) .* VI .+ p.ω .* Vnew.VI
        VC_new = (1-p.ω) .* VC .+ p.ω .* Vnew.VC
        VR_new = (1-p.ω) .* VR .+ p.ω .* Vnew.VR

        # convergence check (undamped error)
        err = maximum((
            maximum(abs.(Vnew.VS .- VS)),
            maximum(abs.(Vnew.VI .- VI)),
            maximum(abs.(Vnew.VC .- VC)),
            maximum(abs.(Vnew.VR .- VR))
        ))

        if !isfinite(err)
            error("Non-finite HJB error at iter=$it (w=$w_fixed).")
        end

        if p.verbose && (it % 50 == 0 || it == 1)
            step = maximum((
                maximum(abs.(VS_new .- VS)),
                maximum(abs.(VI_new .- VI)),
                maximum(abs.(VC_new .- VC)),
                maximum(abs.(VR_new .- VR))
            ))
            println("iter = $it, error = $err, step = $step, ω = $(p.ω), w = $w_fixed")
        end

        # update
        VS .= VS_new
        VI .= VI_new
        VC .= VC_new
        VR .= VR_new

        if err < p.tolHJBvalue
            p.verbose && println("Converged in $it iterations (error = $err)")
            return (VS = VS, VI = VI, VC = VC, VR = VR)
        end
    end

    error("Value iteration did not converge in $(p.maxitHJBvalue) iterations.")
end

mutable struct DerivDkCache{T}
    ∂VS::Vector{T}
    ∂VI::Vector{T}
    ∂VC::Vector{T}
    ∂VR::Vector{T}
end

DerivDkCache(::Type{T}, Nk::Int) where {T} = DerivDkCache(zeros(T, Nk), zeros(T, Nk), zeros(T, Nk), zeros(T, Nk))

mutable struct HJBAssemblyCache{T}
    I::Vector{Int}
    J::Vector{Int}
    X::Vector{T}
    rhs::Vector{T}
end

function HJBAssemblyCache(::Type{T}, Nk::Int) where {T}
    n = 4 * Nk
    return HJBAssemblyCache(Int[], Int[], T[], zeros(T, n))
end

function compute_∂V_dk!(cache::DerivDkCache, V, p)
    ∂k_safe!(cache.∂VS, V.VS, p.Nk, p.Δk, p.ϵDkUp)
    ∂k_safe!(cache.∂VI, V.VI, p.Nk, p.Δk, p.ϵDkUp)
    ∂k_safe!(cache.∂VC, V.VC, p.Nk, p.Δk, p.ϵDkUp)
    ∂k_safe!(cache.∂VR, V.VR, p.Nk, p.Δk, p.ϵDkUp)
    return (∂kVS=cache.∂VS, ∂kVI=cache.∂VI, ∂kVC=cache.∂VC, ∂kVR=cache.∂VR)
end

function compute_∂V_dk(V, p)
    ∂VS_k = ∂k_safe(V.VS, p.Nk, p.Δk, p.ϵDkUp)
    ∂VI_k = ∂k_safe(V.VI, p.Nk, p.Δk, p.ϵDkUp)
    ∂VC_k = ∂k_safe(V.VC, p.Nk, p.Δk, p.ϵDkUp)
    ∂VR_k = ∂k_safe(V.VR, p.Nk, p.Δk, p.ϵDkUp)
    return (∂kVS=∂VS_k, ∂kVI=∂VI_k, ∂kVC=∂VC_k, ∂kVR=∂VR_k)
end

function compute_labor_and_aggregates(V, ∂V, Ft, p; w, K=nothing)
    lOpt, W = optimal_labor_ALL(V, ∂V, Ft, w, p)
    Kval = isnothing(K) ? aggregate_kapital(Ft, p) : K
    L = aggregate_labor_supply(lOpt, Ft, p)
    r = returns(Kval, L, p)
    w_update = wage(Kval, L, p)
    LI = sum(lOpt.lI .* Ft.ϕIt) * p.Δk
    return (lOpt=lOpt, W=W, K=Kval, L=L, r=r, w_update=w_update, LI=LI)
end

function value_iterationHJB(V0, Ft, p)
    """Outer fixed point over wage; inner value iteration converges V given w."""

    V = (VS = copy(V0.VS), VI = copy(V0.VI), VC = copy(V0.VC), VR = copy(V0.VR))
    w = p.w_start
    K = aggregate_kapital(Ft, p)

    dcache = DerivDkCache(eltype(V.VS), p.Nk)

    for itw in 1:p.maxitWage

        p.verbose && println("\nWage iteration itw=$itw, w=$w")

        # 1) Solve HJB given wage
        V = value_iterationHJB_given_wage(V, Ft, p; w = w)

        # 2) Update wage using implied aggregate wage mapping
        ∂V = compute_∂V_dk!(dcache, V, p)
        agg = compute_labor_and_aggregates(V, ∂V, Ft, p; w=w, K=K)
        lOpt = agg.lOpt
        L = agg.L
        w_implied = agg.w_update

        if !(isfinite(w_implied) && w_implied > 0.0)
            error("Implied wage is non-finite or non-positive: w_implied=$w_implied")
        end

        if p.verbose && (itw == 1 || itw % 10 == 0)
            LS = sum(lOpt.lS .* Ft.ϕSt) * p.Δk
            LI = sum(lOpt.lI .* Ft.ϕIt) * p.Δk
            LC = sum(lOpt.lC .* Ft.ϕCt) * p.Δk
            LR = sum(lOpt.lR .* Ft.ϕRt) * p.Δk
            println("diag: K=$K, L=$L (Ls=$(max(L, p.ϵDkUp))), LS=$LS, LI=$LI, LC=$LC, LR=$LR")
            println("diag: lS∈[$(minimum(lOpt.lS)),$(maximum(lOpt.lS))], lI∈[$(minimum(lOpt.lI)),$(maximum(lOpt.lI))], lR∈[$(minimum(lOpt.lR)),$(maximum(lOpt.lR))]")
        end

        gap = abs(w_implied - w)
        p.verbose && println("w_implied=$w_implied gap=$gap ωw=$(p.ωw)")
        if gap < p.tolWage
            return (VS = V.VS, VI = V.VI, VC = V.VC, VR = V.VR, w = w_implied)
        end

        w = (1 - p.ωw) * w + p.ωw * w_implied
    end

    error("Wage fixed point did not converge in $(p.maxitWage) iterations.")
end

# Build sparse linear system: (ρI - A - Q) V = u
function build_HJB_linear_system(V, Ft, p; w0, deriv_cache=nothing, assembly_cache=nothing)

    # Unpack value functions
    VS = V.VS; VI = V.VI; VC = V.VC; VR = V.VR;

    # Compute derivatives V'(k) with a positive floor for numerical stability.
    ∂V = isnothing(deriv_cache) ? compute_∂V_dk(V, p) : compute_∂V_dk!(deriv_cache, V, p)
    ∂VS_k = ∂V.∂kVS
    ∂VI_k = ∂V.∂kVI
    ∂VC_k = ∂V.∂kVC
    ∂VR_k = ∂V.∂kVR

    # Use the provided wage guess inside the HJB operator; update it outside with damping.
    w = isfinite(w0) ? max(w0, p.ϵDkUp) : max(p.w_start, p.ϵDkUp)

    agg = compute_labor_and_aggregates(V, ∂V, Ft, p; w=w)
    lOpt = agg.lOpt
    W = agg.W
    K = agg.K
    L = agg.L
    r = agg.r
    w_update = agg.w_update
    LI = agg.LI
    if p.verbose
        if !all(isfinite, W.WS)
            error("Non-finite values in WS (effective wage for S).")
        end
        if !(isfinite(W.WI) && isfinite(W.WC) && isfinite(W.WR))
            error("Non-finite effective wage scalar (WI/WC/WR). WI=$(W.WI) WC=$(W.WC) WR=$(W.WR)")
        end
    end

    if p.verbose && !(isfinite(K) && isfinite(L) && isfinite(r))
        error("Non-finite aggregates: K=$K L=$L r=$r")
    end

    # capial income for each capital level
    capital_income = (r - p.δ) .* p.k

    # Optimal consumption from FOC: u_c = θ/c = V'(k)  =>  c = θ / V'(k)
    cS = p.θ ./ ∂VS_k
    cI = p.θ ./ ∂VI_k
    cC = p.θ ./ ∂VC_k
    cR = p.θ ./ ∂VR_k

    # State-constraints at the boundaries must be enforced on controls (not just by clipping drift).
    # At k = 0, require b >= 0  =>  c <= income.
    # At k = kmax, require b <= 0  =>  c >= income.
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

    Nk = p.Nk
    Δk = p.Δk

    # Rates (k-dependent)
    infection_rate = p.β .* lOpt.lS .* LI

    v_rate = clamp.((VR .- VS) ./ p.γ, 0.0, p.qMax)
    v_cost = -0.5 .* p.γ .* (v_rate .^ 2)

    # Flow utilities u(k): compute from original utility using continuous controls
    epsu = p.ϵDkUp
    uS = p.θ .* log.(max.(cS, epsu)) .+ (1 - p.θ) .* log.(max.(1 .- lOpt.lS, epsu)) .+ v_cost
    uI = p.θ .* log.(max.(cI, epsu)) .+ (1 - p.θ) .* log.(max.(1 .- lOpt.lI, epsu)) .- p.dI
    uC = p.θ .* log.(max.(cC, epsu)) .- p.dC
    uR = p.θ .* log.(max.(cR, epsu)) .+ (1 - p.θ) .* log.(max.(1 .- lOpt.lR, epsu))

    # Drifts b(k) = capital income + labor income - consumption
    bS = incomeS .- cS
    bI = incomeI .- cI
    bC = incomeC .- cC
    bR = incomeR .- cR

    # State constraints at boundaries: no drift leaving the domain
    bS[1] = max(bS[1], 0.0); bS[end] = min(bS[end], 0.0)
    bI[1] = max(bI[1], 0.0); bI[end] = min(bI[end], 0.0)
    bC[1] = max(bC[1], 0.0); bC[end] = min(bC[end], 0.0)
    bR[1] = max(bR[1], 0.0); bR[end] = min(bR[end], 0.0)

    n = 4 * Nk
    I = Int[]
    J = Int[]
    X = eltype(VS)[]
    rhs = zeros(eltype(VS), n)

    if !isnothing(assembly_cache)
        I = assembly_cache.I
        J = assembly_cache.J
        X = assembly_cache.X
        rhs = assembly_cache.rhs
        empty!(I); empty!(J); empty!(X)
        fill!(rhs, 0)
    end

    sizehint!(I, 8n)
    sizehint!(J, 8n)
    sizehint!(X, 8n)

    idx(state, i) = (state - 1) * Nk + i

    function push_entry!(row, col, val)
        push!(I, row)
        push!(J, col)
        push!(X, val)
        return nothing
    end

    # Upwind (Godunov/flux-splitting) discretization for the drift term.
    # Given b_i (drift at node i) this adds the tri-diagonal entries corresponding to
    # a first-order upwind approximation based on b_i^+ and b_i^-.
    function add_upwind_drift_entries!(row, i, b_i)
        bplus = max(b_i, 0.0)
        bminus = min(b_i, 0.0)
        aL = bplus / Δk
        aU = (-bminus) / Δk

        push_entry!(row, row, aL + aU)
        if i > 1
            push_entry!(row, row - 1, -aL)
        end
        if i < Nk
            push_entry!(row, row + 1, -aU)
        end
        return nothing
    end

    exitI = p.σ1 + p.σ3 + p.μ
    exitC = (p.αEpi + p.μ) + p.σ2
    exitR = p.λ + p.μ

    for i in 1:Nk
        # S row
        rowS = idx(1, i)
        outS = infection_rate[i] + v_rate[i]
        push_entry!(rowS, rowS, p.ρ + outS)
        add_upwind_drift_entries!(rowS, i, -bS[i])
        push_entry!(rowS, idx(2, i), -infection_rate[i])
        push_entry!(rowS, idx(4, i), -v_rate[i])
        rhs[rowS] = uS[i]

        # I row
        rowI = idx(2, i)
        push_entry!(rowI, rowI, p.ρ + exitI)
        add_upwind_drift_entries!(rowI, i, -bI[i])
        push_entry!(rowI, idx(1, i), -p.μ)
        push_entry!(rowI, idx(3, i), -p.σ1)
        push_entry!(rowI, idx(4, i), -p.σ3)
        rhs[rowI] = uI[i]

        # C row
        rowC = idx(3, i)
        push_entry!(rowC, rowC, p.ρ + exitC)
        add_upwind_drift_entries!(rowC, i, -bC[i])
        push_entry!(rowC, idx(1, i), -(p.αEpi + p.μ))
        push_entry!(rowC, idx(4, i), -p.σ2)
        rhs[rowC] = uC[i]

        # R row
        rowR = idx(4, i)
        push_entry!(rowR, rowR, p.ρ + exitR)
        add_upwind_drift_entries!(rowR, i, -bR[i])
        push_entry!(rowR, idx(1, i), -exitR)
        rhs[rowR] = uR[i]
    end

    M = sparse(I, J, X, n, n)
    return (M=M, rhs=rhs, w_update=w_update, w_used=w, Nk=Nk)
end

# HJB operator T
function T_HJB(V, Ft, p; w0, deriv_cache=nothing, assembly_cache=nothing)

    sys = build_HJB_linear_system(V, Ft, p; w0=w0, deriv_cache=deriv_cache, assembly_cache=assembly_cache)

    Nk = sys.Nk
    Vvec = sys.M \ sys.rhs
    VS_new = Vvec[1:Nk]
    VI_new = Vvec[Nk+1:2Nk]
    VC_new = Vvec[2Nk+1:3Nk]
    VR_new = Vvec[3Nk+1:4Nk]

    if p.verbose && !(all(isfinite, VS_new) && all(isfinite, VI_new) && all(isfinite, VC_new) && all(isfinite, VR_new))
        error("Non-finite values in implicit HJB solve (w=$(sys.w_used)).")
    end

    return (VS = VS_new, VI = VI_new, VC = VC_new, VR = VR_new), sys.w_update
end


