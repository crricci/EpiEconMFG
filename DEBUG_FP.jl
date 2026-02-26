function create_test_distribution(p)
    St = 0.7 * ones(p.Nk)
    It = 0.1 * ones(p.Nk)
    Ct = 0.1 * ones(p.Nk)
    Rt = 0.1 * ones(p.Nk)
    Mass = sum(St + It + Ct + Rt) * p.Δk
    St .= St ./ Mass
    It .= It ./ Mass
    Ct .= Ct ./ Mass
    Rt .= Rt ./ Mass
    return (ϕSt = St, ϕIt = It, ϕCt = Ct, ϕRt = Rt)
end

include("L_loadAll.jl")

p = MFGEpiEcon(ω=0.1, maxitHJBvalue=3000, ωw=0.2, maxitWage=200, tolWage=1e-3, verbose=true, Δt=0.05, HJB_every=5)

F0 = create_test_distribution(p)
V0 = (
    VS = 0.4 * sqrt.(p.k),
    VI = 0.3 * sqrt.(p.k),
    VC = 0.2 * sqrt.(p.k),
    VR = 0.5 * sqrt.(p.k),
)

Nstep_eff = Int(ceil(p.T_End / p.Δt))
println("DEBUG_FP: Nk=$(p.Nk), Δk=$(p.Δk), T_End=$(p.T_End), Δt=$(p.Δt), Nstep≈$Nstep_eff, HJB_every=$(p.HJB_every)")

try
    t, Fts, prices = simulate_FP(F0, V0, p)

    Fend = Fts[end]
    mass_end = sum(Fend.ϕSt .+ Fend.ϕIt .+ Fend.ϕCt .+ Fend.ϕRt) * p.Δk
    println("DEBUG_FP: mass_end=$mass_end (deviation=$(mass_end - 1.0))")

    I_mass0 = sum(F0.ϕIt) * p.Δk
    I_massT = sum(Fend.ϕIt) * p.Δk
    println("DEBUG_FP: infected mass I(t=0)=$I_mass0, I(t=T)=$I_massT")

    println("DEBUG_FP: last prices w=$(prices.w[end]) r=$(prices.r[end]) K=$(prices.K[end]) L=$(prices.L[end]) LI=$(prices.LI[end])")
catch e
    println("DEBUG_FP failed: $e")
    rethrow(e)
end
