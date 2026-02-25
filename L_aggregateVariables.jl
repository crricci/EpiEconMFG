
function optimal_labor(∂kV, W, p)
    lOpt = @. max(0.0, 1.0 - (1.0 - p.θ) / (∂kV * W))
    return lOpt
end

function optimal_labor_ALL(V, ∂V, F, w, p)

    # compute first WI using the current guess w
    WI = p.ηI * w
    lOpt_I = optimal_labor(∂V.∂kVI, WI, p)

    # then compute LI which depends on lOpt_I and the value functions 
    LI = sum(lOpt_I .* F.ϕIt) * p.Δk  

    # then compute WS which depends on WI and the value functions
    WS = p.ηS * w .+ p.β * LI .* (V.VI .- V.VS) ./ ∂V.∂kVS  
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
    k = collect(p.k)

    KS = sum(k .* ϕSt) * p.Δk  
    KI = sum(k .* ϕIt) * p.Δk  
    KC = sum(k .* ϕCt) * p.Δk  
    KR = sum(k .* ϕRt) * p.Δk  

    K = KS + KI + KC + KR
    return K
end