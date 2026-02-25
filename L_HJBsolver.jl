
function value_iterationHJB_given_wage(V0, Ft, p; w)
    """
    V0: NamedTuple
        Initial guess (VS, VI, VC, VR), each Vector{Float64}
    Ft: NamedTuple
        Fokker-Planck solution (ϕSt, ϕIt, ϕCt, ϕRt), each a Vector{Float64} Nk

    p: model parameters (MFGEpiEcon)
    """
    # current iterate
    VS = copy(V0.VS)
    VI = copy(V0.VI)
    VC = copy(V0.VC)
    VR = copy(V0.VR)

    w_fixed = w

    for it in 1:p.maxitHJBvalue

        # HJB operator with fixed wage
        Vnew, _ = T_HJB((VS = VS, VI = VI, VC = VC, VR = VR), Ft, p; w0 = w_fixed)

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

function value_iterationHJB(V0, Ft, p)
    """Outer fixed point over wage; inner value iteration converges V given w."""

    V = (VS = copy(V0.VS), VI = copy(V0.VI), VC = copy(V0.VC), VR = copy(V0.VR))
    w = p.w_start

    for itw in 1:p.maxitWage

        p.verbose && println("\nWage iteration itw=$itw, w=$w")

        # 1) Solve HJB given wage
        V = value_iterationHJB_given_wage(V, Ft, p; w = w)

        # 2) Update wage using implied aggregate wage mapping
        ∂VS_log = ∂k_log(V.VS, p.Nk, p.Δk, p.ϵDkUp)
        ∂VI_log = ∂k_log(V.VI, p.Nk, p.Δk, p.ϵDkUp)
        ∂VC_log = ∂k_log(V.VC, p.Nk, p.Δk, p.ϵDkUp)
        ∂VR_log = ∂k_log(V.VR, p.Nk, p.Δk, p.ϵDkUp)

        # Implied wage mapping w -> w' using current policies and exogenous distribution
        ∂V = (∂kVS=∂VS_log, ∂kVI=∂VI_log, ∂kVC=∂VC_log, ∂kVR=∂VR_log)
        lOpt, _ = optimal_labor_ALL(V, ∂V, Ft, w, p)
        K = aggregate_kapital(Ft, p)
        L = aggregate_labor_supply(lOpt, Ft, p)
        w_implied = wage(K, L, p)

        if !(isfinite(w_implied) && w_implied > 0.0)
            error("Implied wage is non-finite or non-positive: w_implied=$w_implied")
        end

        if p.verbose
            LS = sum(lOpt.lS .* Ft.ϕSt) * p.Δk
            LI = sum(lOpt.lI .* Ft.ϕIt) * p.Δk
            LC = sum(lOpt.lC .* Ft.ϕCt) * p.Δk
            LR = sum(lOpt.lR .* Ft.ϕRt) * p.Δk
            Ls = max(L, p.ϵDkUp)
            println("diag: K=$K, L=$L (Ls=$Ls), LS=$LS, LI=$LI, LC=$LC, LR=$LR")
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

# HJB operator T
function T_HJB(V, Ft, p; w0)


    # Unpack value functions
    VS = V.VS; VI = V.VI; VC = V.VC; VR = V.VR;

    # Compute derivatives V'(k). Safe log-derivative
    ∂VS_log = ∂k_log(VS, p.Nk, p.Δk, p.ϵDkUp)
    ∂VI_log = ∂k_log(VI, p.Nk, p.Δk, p.ϵDkUp)
    ∂VC_log = ∂k_log(VC, p.Nk, p.Δk, p.ϵDkUp)
    ∂VR_log = ∂k_log(VR, p.Nk, p.Δk, p.ϵDkUp)   

    # Use the provided wage guess inside the HJB operator; update it outside with damping.
    w = isfinite(w0) ? max(w0, p.ϵDkUp) : max(p.w_start, p.ϵDkUp)

    # effective wages and optimal labor supply for each health state
    lOpt, W = optimal_labor_ALL(V, (∂kVS=∂VS_log, ∂kVI=∂VI_log, ∂kVC=∂VC_log, ∂kVR=∂VR_log), Ft, w, p)
    LI =sum(lOpt.lI .* Ft.ϕIt) * p.Δk           # for transition from S to I
    WS = W.WS; WI = W.WI; WC = W.WC; WR = W.WR;
    # side note: WS is an array. WI, WC, WR are scalars (independent of k)

    if p.verbose
        if !all(isfinite, WS)
            error("Non-finite values in WS (effective wage for S).")
        end
        if !(isfinite(WI) && isfinite(WC) && isfinite(WR))
            error("Non-finite effective wage scalar (WI/WC/WR). WI=$WI WC=$WC WR=$WR")
        end
    end
    
    # returns to capital (interest rate)
    K = aggregate_kapital(Ft, p)
    L = aggregate_labor_supply(lOpt, Ft, p)
    r = returns(K, L, p)
    w_update = wage(K, L, p)

    if p.verbose && !(isfinite(K) && isfinite(L) && isfinite(r))
        error("Non-finite aggregates: K=$K L=$L r=$r")
    end

    # capial income for each capital level
    k = collect(p.k)
    capital_income = (r - p.δ) * k 

    # Optimal consumption from FOC: u_c = θ/c = V'(k)  =>  c = θ / V'(k)
    cS = p.θ ./ ∂VS_log
    cI = p.θ ./ ∂VI_log
    cC = p.θ ./ ∂VC_log
    cR = p.θ ./ ∂VR_log

    # Drift b(k) = capital income + labor income - consumption.
    # We compute separate drifts for work vs no-work regimes (labor income depends on l).
    bS_work = capital_income .+ (p.ηS * w) .* lOpt.lS .- cS
    bI_work = capital_income .+ (p.ηI * w) .* lOpt.lI .- cI
    bR_work = capital_income .+ (p.ηR * w) .* lOpt.lR .- cR

    bS_nowork = capital_income .- cS
    bI_nowork = capital_income .- cI
    bC_nowork = capital_income .- cC
    bR_nowork = capital_income .- cR

    # Upwind derivatives V'(k) selected based on drift sign
    ∂VS_flux_Work = ∂k_flux(VS, bS_work, p)
    ∂VI_flux_Work = ∂k_flux(VI, bI_work, p)
    ∂VR_flux_Work = ∂k_flux(VR, bR_work, p)

    ∂VS_flux_NoWork = ∂k_flux(VS, bS_nowork, p)
    ∂VI_flux_NoWork = ∂k_flux(VI, bI_nowork, p)
    ∂VC_flux_NoWork = ∂k_flux(VC, bC_nowork, p)
    ∂VR_flux_NoWork = ∂k_flux(VR, bR_nowork, p)

    
    # Utility constants
    Ū = p.θ * log(p.θ) + (1 - p.θ) * log(1 - p.θ) - 1
    Û = p.θ * log(p.θ) - p.θ

    Nk = p.Nk
    Δk = p.Δk

    # Rates (k-dependent)
    infection_rate = p.β .* lOpt.lS .* LI

    v_rate = clamp.((VR .- VS) ./ p.γ, 0.0, 1.0)
    v_cost = -0.5 .* p.γ .* (v_rate .^ 2)

    # Flow utilities u(k): compute from original utility using continuous controls
    epsu = p.ϵDkUp
    uS = p.θ .* log.(max.(cS, epsu)) .+ (1 - p.θ) .* log.(max.(1 .- lOpt.lS, epsu)) .+ v_cost
    uI = p.θ .* log.(max.(cI, epsu)) .+ (1 - p.θ) .* log.(max.(1 .- lOpt.lI, epsu)) .- p.dI
    uC = p.θ .* log.(max.(cC, epsu)) .- p.dC
    uR = p.θ .* log.(max.(cR, epsu)) .+ (1 - p.θ) .* log.(max.(1 .- lOpt.lR, epsu))

    # Drifts b(k) = capital income + labor income - consumption
    bS = capital_income .+ (p.ηS * w) .* lOpt.lS .- cS
    bI = capital_income .+ (p.ηI * w) .* lOpt.lI .- cI
    bC = capital_income .- cC
    bR = capital_income .+ (p.ηR * w) .* lOpt.lR .- cR

    # State constraints at boundaries: no drift leaving the domain
    bS[1] = max(bS[1], 0.0); bS[end] = min(bS[end], 0.0)
    bI[1] = max(bI[1], 0.0); bI[end] = min(bI[end], 0.0)
    bC[1] = max(bC[1], 0.0); bC[end] = min(bC[end], 0.0)
    bR[1] = max(bR[1], 0.0); bR[end] = min(bR[end], 0.0)

    # Build sparse linear system: (ρI - A - Q) V = u
    n = 4 * Nk
    M = spzeros(n, n)
    rhs = zeros(n)

    idx(state, i) = (state - 1) * Nk + i

    function add_drift_row!(row, i, b)
        b_i = b[i]
        bplus = max(b_i, 0.0)
        bminus = min(b_i, 0.0)
        aL = bplus / Δk
        aU = (-bminus) / Δk

        M[row, row] += aL + aU
        if i > 1
            M[row, row - 1] += -aL
        end
        if i < Nk
            M[row, row + 1] += -aU
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
        M[rowS, rowS] += p.ρ + outS
        add_drift_row!(rowS, i, bS)
        M[rowS, idx(2, i)] += -infection_rate[i]
        M[rowS, idx(4, i)] += -v_rate[i]
        rhs[rowS] = uS[i]

        # I row
        rowI = idx(2, i)
        M[rowI, rowI] += p.ρ + exitI
        add_drift_row!(rowI, i, bI)
        M[rowI, idx(1, i)] += -p.μ
        M[rowI, idx(3, i)] += -p.σ1
        M[rowI, idx(4, i)] += -p.σ3
        rhs[rowI] = uI[i]

        # C row
        rowC = idx(3, i)
        M[rowC, rowC] += p.ρ + exitC
        add_drift_row!(rowC, i, bC)
        M[rowC, idx(1, i)] += -(p.αEpi + p.μ)
        M[rowC, idx(4, i)] += -p.σ2
        rhs[rowC] = uC[i]

        # R row
        rowR = idx(4, i)
        M[rowR, rowR] += p.ρ + exitR
        add_drift_row!(rowR, i, bR)
        M[rowR, idx(1, i)] += -exitR
        rhs[rowR] = uR[i]
    end

    Vvec = M \ rhs
    VS_new = Vvec[1:Nk]
    VI_new = Vvec[Nk+1:2Nk]
    VC_new = Vvec[2Nk+1:3Nk]
    VR_new = Vvec[3Nk+1:4Nk]

    if p.verbose && !(all(isfinite, VS_new) && all(isfinite, VI_new) && all(isfinite, VC_new) && all(isfinite, VR_new))
        error("Non-finite values in implicit HJB solve (w=$w).")
    end

    return (VS = VS_new, VI = VI_new, VC = VC_new, VR = VR_new), w_update
end


