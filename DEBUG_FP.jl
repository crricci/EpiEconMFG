include("L_loadAll.jl")
p = MFGEpiEcon(verbose=true)

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
