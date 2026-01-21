

w = wFun(Ft,p)
r = rFun(Ft,p)
LII = LIIFun(Ft,p)


function optimal_labor(∂kV, W, p)
    lOpt = @. max(0.0, 1.0 - (1.0 - p.θ) / (∂kV * W))
    return lOpt
end

function optimal_labor_ALL(∂kV, W, p)
    lOpt_S = optimal_labor(∂kV.∂kVS, W.WS, p)
    lOpt_I = optimal_labor(∂kV.∂kVI, W.WI, p)
    lOpt_C = optimal_labor(∂kV.∂kVC, W.WC, p)
    lOpt_R = optimal_labor(∂kV.∂kVR, W.WR, p)
    return (lOpt_S = lOpt_S, lOpt_I = lOpt_I, lOpt_C = lOpt_C, lOpt_R = lOpt_R)
end

function aggregate_labor_supply(lOpt, Ft, p)

    lOpt_S = lOpt.lOpt_S; lOpt_I = lOpt.lOpt_I; lOpt_C = lOpt.lOpt_C; lOpt_R = lOpt.lOpt_R;
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