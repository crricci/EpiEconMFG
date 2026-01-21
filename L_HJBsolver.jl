
function value_iterationHJB(V0, p, Ft)
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

    for it in 1:p.maxitHJBvalue

        # HJB operator
        Vnew = T_HJB((VS = VS, VI = VI, VC = VC, VR = VR), p, Ft)

        # damped update
        VS_new = (1-p.ω) .* VS .+ p.ω .* Vnew.VS
        VI_new = (1-p.ω) .* VI .+ p.ω .* Vnew.VI
        VC_new = (1-p.ω) .* VC .+ p.ω .* Vnew.VC
        VR_new = (1-p.ω) .* VR .+ p.ω .* Vnew.VR

        # convergence check
        err = maximum((
            maximum(abs.(VS_new .- VS)),
            maximum(abs.(VI_new .- VI)),
            maximum(abs.(VC_new .- VC)),
            maximum(abs.(VR_new .- VR))
        ))

        if p.verbose && (it % 50 == 0 || it == 1)
            println("iter = $it, error = $err")
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

# HJB operator T
function T_HJB(V, p, Ft)


    # Unpack value functions
    VS = V.VS; VI = V.VI; VC = V.VC; VR = V.VR;

    # Compute derivatives V'(k). Safe log-derivative
    ∂VS_log = ∂k_log(VS, p.Nk, p.Δk, p.ϵDkUp)
    ∂VI_log = ∂k_log(VI, p.Nk, p.Δk, p.ϵDkUp)
    ∂VC_log = ∂k_log(VC, p.Nk, p.Δk, p.ϵDkUp)
    ∂VR_log = ∂k_log(VR, p.Nk, p.Δk, p.ϵDkUp)   

    # Compute wages for each state (proportional to productivity times equilibrium wage)
    WS = p.ηS * w .+ p.β * LI .* (VI .- VS) ./ ∂VS_log  # Effective wage including infection risk compensation
    WI = p.ηI * w
    WC = p.ηC * w  # Zero for contained, useless
    WR = p.ηR * w
    
    lOpt = optimal_labor_ALL((∂kVS=∂VS_log, ∂kVI=∂VI_log, ∂kVC=∂VC_log, ∂kVR=∂VR_log), (WS=WS, WI=WI, WC=WC, WR=WR), p)
    LI = sum(lOpt.lI .* Ft.ϕIt) * p.Δk  
    L = aggregate_labor_supply(lOpt, Ft, p)
    K = aggregate_kapital(Ft, p)
    w = wage(K,L,p)
    r = returns(K,L,p)


    # Capital accumulation term
    k = collect(p.k)
    capital_income = (r - p.δ) * k 
    

    # Compute derivatives V'(k), different upwind scheme depending on the drift
    # in case they work
    ∂VS_flux_Work = ∂k_flux(VS, capital_income + p.ηS * w, p)
    ∂VI_flux_Work = ∂k_flux(VI, capital_income + p.ηI * w, p)
    ∂VC_flux_Work = ∂k_flux(VC, capital_income + p.ηC * w, p)
    ∂VR_flux_Work = ∂k_flux(VR, capital_income + p.ηR * w, p)

    # in case they do not work
    ∂VS_flux_NoWork = ∂k_flux(VS, capital_income, p)
    ∂VI_flux_NoWork = ∂k_flux(VI, capital_income, p)
    ∂VC_flux_NoWork = ∂k_flux(VC, capital_income, p)
    ∂VR_flux_NoWork = ∂k_flux(VR, capital_income, p)

    

    
    # Utility constants
    Ū = p.θ * log(p.θ) + (1 - p.θ) * log(1 - p.θ) - 1
    Û = p.θ * log(p.θ) - p.θ
    
    # Initialize new value functions
    VS_new = similar(VS)
    VI_new = similar(VI)
    VC_new = similar(VC)
    VR_new = similar(VR)
    
    # HJB for Susceptible (S)
    for i in 1:p.Nk
        vaccination_benefit = (VR[i] - VS[i])^2 / (2 * p.γ)
        transition_S_to_I = p.β * LI * (VI[i] - VS[i])
        
        # Check if agent works: V'(k) * W(k) > 1-θ
        if ∂VS_log[i] * WS[i] > (1 - p.θ)
            # Agent works
            flow_utility = Ū - p.θ * log(∂VS_log[i]) - (1 - p.θ) * log(max(∂VS_log[i] * p.ηS * w + transition_S_to_I,p.ϵDkUp))
            capital_term = ∂VS_flux_Work[i] * (capital_income[i] + p.ηS * w)
        else
            # Agent doesn't work
            flow_utility = Û - p.θ * log(∂VS_log[i])
            capital_term = ∂VS_flux_NoWork[i] * capital_income[i]
        end
        
        VS_new[i] = (flow_utility + capital_term + vaccination_benefit + transition_S_to_I) / p.ρ
    end
    
    # HJB for Infected (I)
    for i in 1:p.Nk
        transition_I_to_C = (p.σ1 + p.σ3) * (VC[i] - VI[i])
        transition_I_to_S = p.μ * (VS[i] - VI[i])
        
        # Check if agent works: V'(k) * W(k) > 1-θ
        if ∂VI_log[i] * WI > (1 - p.θ)
            # Agent works
            flow_utility = Ū - p.θ * log(∂VI_log[i]) - (1 - p.θ) * log(WI) - p.dI
            capital_term = ∂VI_flux_Work[i] * (capital_income[i] + p.ηI * w)
        else
            # Agent doesn't work
            flow_utility = Û - p.θ * log(∂VI_log[i]) - p.dI
            capital_term = ∂VI_flux_NoWork[i] * capital_income[i]
        end
        
        VI_new[i] = (flow_utility + capital_term + transition_I_to_C + transition_I_to_S) / p.ρ
    end
    
    # HJB for Contained (C)
    # Contained agents never work
    for i in 1:p.Nk
        flow_utility = Û - p.θ * log(∂VC_log[i]) - p.dC
        capital_term = ∂VC_flux_NoWork[i] * capital_income[i]
        transition_C_to_S = (p.α + p.μ) * (VS[i] - VC[i])
        transition_C_to_R = p.σ2 * (VR[i] - VC[i])
        
        VC_new[i] = (flow_utility + capital_term + transition_C_to_S + transition_C_to_R) / p.ρ
    end
    
    # HJB for Recovered (R)
    for i in 1:p.Nk
        transition_R_to_S = (p.λ + p.μ) * (VS[i] - VR[i])
        
        # Check if agent works: V'(k) * W(k) > 1-θ
        if ∂VR_log[i] * WR > (1 - p.θ)
            # Agent works
            flow_utility = Ū - p.θ * log(∂VR_log[i]) - (1 - p.θ) * log(WR)
            capital_term = ∂VR_flux_Work[i] * (capital_income[i] + p.ηR * w)
        else
            # Agent doesn't work
            flow_utility = Û - p.θ * log(∂VR_log[i])
            capital_term = ∂VR_flux_NoWork[i] * capital_income[i]
        end
        
        VR_new[i] = (flow_utility + capital_term + transition_R_to_S) / p.ρ
    end
    
    return (VS = VS_new, VI = VI_new, VC = VC_new, VR = VR_new)
end


