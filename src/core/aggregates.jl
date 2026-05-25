
function optimal_labor(∂kV, W, p)
    # If effective wage is non-positive, the agent never works.
    # Labor is constrained to [0,1].
    lRaw = @. ifelse(W <= 0.0, 0.0, (1.0 - (1.0 - p.θ) / (∂kV * W)))
    return @. min(1.0, max(0.0, lRaw))
end

function optimal_labor_ALL(V, ∂V, F, w, p)

    # compute first WI using the current guess w
    WI = p.ηI * w
    lOpt_I = optimal_labor(∂V.∂kVI, WI, p)

    # then compute LI which depends on lOpt_I and the value functions 
    LI = sum(lOpt_I .* F.ϕIt) * p.Δk  

    # then compute WS which depends on WI and the value functions
    # Avoid dividing by the very small floor used for log-derivatives.
    denomS = max.(∂V.∂kVS, sqrt(p.ϵDkUp))
    WS = p.ηS * w .+ p.β * LI .* (V.VI .- V.VS) ./ denomS
    WC = p.ηC * w  # Zero for contained, useless
    WR = p.ηR * w

    lOpt_S = optimal_labor(∂V.∂kVS, WS, p)
    lOpt_C = optimal_labor(∂V.∂kVC, WC, p)
    lOpt_R = optimal_labor(∂V.∂kVR, WR, p)

    return (lS = lOpt_S, lI = lOpt_I, lC = lOpt_C, lR = lOpt_R), (WS = WS, WI = WI, WC = WC, WR = WR)
end

function aggregate_labor_supply(lOpt, Ft, p)

    lOpt_S = lOpt.lS; lOpt_I = lOpt.lI; lOpt_C = lOpt.lC; lOpt_R = lOpt.lR;
    ϕSt = Ft.ϕSt; ϕIt = Ft.ϕIt; ϕCt = Ft.ϕCt; ϕRt = Ft.ϕRt;

    LS = sum(lOpt_S .* ϕSt) * p.Δk  
    LI = sum(lOpt_I .* ϕIt) * p.Δk  
    LC = sum(lOpt_C .* ϕCt) * p.Δk  
    LR = sum(lOpt_R .* ϕRt) * p.Δk  

    L = p.ηS * LS + p.ηI * LI + p.ηC * LC + p.ηR * LR
    return L
end

function aggregate_kapital(Ft, p)

    ϕSt = Ft.ϕSt; ϕIt = Ft.ϕIt; ϕCt = Ft.ϕCt; ϕRt = Ft.ϕRt;
    k = p.k

    KS = sum(k .* ϕSt) * p.Δk  
    KI = sum(k .* ϕIt) * p.Δk  
    KC = sum(k .* ϕCt) * p.Δk  
    KR = sum(k .* ϕRt) * p.Δk  

    K = KS + KI + KC + KR
    return K
end

function infected_labor_supply(lOpt, Ft, p)
    return sum(lOpt.lI .* Ft.ϕIt) * p.Δk
end

function aggregate_labor_supply_path(F_path, controls_path, p)
    Nt = length(F_path)
    L = Vector{Float64}(undef, Nt)
    @inbounds for n in 1:Nt
        L[n] = aggregate_labor_supply(controls_path[n].lOpt, F_path[n], p)
    end
    return L
end

function compute_aggregates_path(F_path, controls_path, p)
    Nt = length(F_path)
    if length(controls_path) != Nt
        error("F_path and controls_path must have the same length")
    end

    K = Vector{Float64}(undef, Nt)
    L = Vector{Float64}(undef, Nt)
    LI = Vector{Float64}(undef, Nt)

    @inbounds for n in 1:Nt
        Ft = F_path[n]
        lOpt = controls_path[n].lOpt
        K[n] = aggregate_kapital(Ft, p)
        L[n] = aggregate_labor_supply(lOpt, Ft, p)
        LI[n] = infected_labor_supply(lOpt, Ft, p)
    end

    return (K = K, L = L, LI = LI)
end

function compute_prices_path(F_path, controls_path, p)
    agg = compute_aggregates_path(F_path, controls_path, p)
    Nt = length(F_path)
    w = Vector{Float64}(undef, Nt)
    r = Vector{Float64}(undef, Nt)

    @inbounds for n in 1:Nt
        w[n] = wage(agg.K[n], agg.L[n], p)
        r[n] = returns(agg.K[n], agg.L[n], p)
    end

    return (w = w, r = r, K = agg.K, L = agg.L, LI = agg.LI)
end
