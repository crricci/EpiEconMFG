

function create_test_distribution(p)
    St = 0.7 * ones(p.Nk) 
    It = 0.1 * ones(p.Nk)
    Ct = 0.1 * ones(p.Nk)
    Rt = 0.1 * ones(p.Nk)
    Mass = sum(St + It + Ct + Rt) *  p.Δk
    St .= St ./ Mass
    It .= It ./ Mass
    Ct .= Ct ./ Mass
    Rt .= Rt ./ Mass
    return (ϕSt = St, ϕIt = It, ϕCt = Ct, ϕRt = Rt)
end


include("L_loadAll.jl")
p = MFGEpiEcon(ω=0.1, maxitHJBvalue=3000, ωw=0.2, maxitWage=200, tolWage=1e-3, verbose=true)
Ft = create_test_distribution(p)

V = (
    VS = 0.4*sqrt.(p.k),
    VI = 0.3*sqrt.(p.k),
    VC = 0.2*sqrt.(p.k),
    VR = 0.5*sqrt.(p.k)
)


Vsol = nothing
try
    println("DEBUG: Nk=$(p.Nk), Δk=$(p.Δk), ω=$(p.ω), tol=$(p.tolHJBvalue), maxit=$(p.maxitHJBvalue)")
    println("DEBUG: w_start=$(p.w_start), ϵDkUp=$(p.ϵDkUp)")
    global Vsol = value_iterationHJB(V, Ft, p)
catch e
    println("value_iterationHJB failed: $e")
end

if Vsol !== nothing
    mass = sum(Ft.ϕSt .+ Ft.ϕIt .+ Ft.ϕCt .+ Ft.ϕRt) * p.Δk
    println("DEBUG: mass(Ft)=$mass (deviation=$(mass - 1.0))")

    # One-step operator residual at the reported solution
    Vcheck, wcheck = T_HJB((VS=Vsol.VS, VI=Vsol.VI, VC=Vsol.VC, VR=Vsol.VR), Ft, p; w0=Vsol.w)
    resS = maximum(abs.(Vcheck.VS .- Vsol.VS))
    resI = maximum(abs.(Vcheck.VI .- Vsol.VI))
    resC = maximum(abs.(Vcheck.VC .- Vsol.VC))
    resR = maximum(abs.(Vcheck.VR .- Vsol.VR))
    println("DEBUG: ||T(V)-V||_∞: S=$resS I=$resI C=$resC R=$resR")

    # Exact linear-system residual at the reported solution: || M*V - rhs ||_∞
    sys = build_HJB_linear_system((VS=Vsol.VS, VI=Vsol.VI, VC=Vsol.VC, VR=Vsol.VR), Ft, p; w0=Vsol.w)
    Vstack = vcat(Vsol.VS, Vsol.VI, Vsol.VC, Vsol.VR)
    lin_res = maximum(abs.(sys.M * Vstack .- sys.rhs))
    println("DEBUG: ||M*V - rhs||_∞ = $lin_res")

    # Wage fixed-point residual at the reported solution
    ∂VS_k = ∂k_safe(Vsol.VS, p.Nk, p.Δk, p.ϵDkUp)
    ∂VI_k = ∂k_safe(Vsol.VI, p.Nk, p.Δk, p.ϵDkUp)
    ∂VC_k = ∂k_safe(Vsol.VC, p.Nk, p.Δk, p.ϵDkUp)
    ∂VR_k = ∂k_safe(Vsol.VR, p.Nk, p.Δk, p.ϵDkUp)
    ∂V = (∂kVS=∂VS_k, ∂kVI=∂VI_k, ∂kVC=∂VC_k, ∂kVR=∂VR_k)
    lOpt, _ = optimal_labor_ALL((VS=Vsol.VS, VI=Vsol.VI, VC=Vsol.VC, VR=Vsol.VR), ∂V, Ft, Vsol.w, p)
    K = aggregate_kapital(Ft, p)
    L = aggregate_labor_supply(lOpt, Ft, p)
    w_implied = wage(K, L, p)
    println("DEBUG: wage residual |w_implied - w| = $(abs(w_implied - Vsol.w)) (w=$(Vsol.w), w_implied=$w_implied, wcheck_from_T=$wcheck)")

    plot(p.k, Vsol.VS, label="VS", xlabel="k", ylabel="Value", title="HJB")
    plot!(p.k, Vsol.VI, label="VI")
    plot!(p.k, Vsol.VC, label="VC")
    plot!(p.k, Vsol.VR, label="VR")

    p.w_start = Vsol.w
    value_iterationHJB((VS=Vsol.VS, VI=Vsol.VI, VC=Vsol.VC, VR=Vsol.VR), Ft, p)  # test warm start with wage from previous solve
end