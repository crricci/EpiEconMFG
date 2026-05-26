using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "main.jl"))


p = EpiEconMFG.MFGEpiEcon(T_End = 4.0, ξ = 0.001)
result = run_dynamic(p=p)
EpiEconMFG.save_all_figures(result,p; outdir=joinpath(@__DIR__,"..","outputs/Dynamic_T4_xi0001"))

