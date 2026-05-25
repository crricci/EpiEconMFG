using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "main.jl"))


p = EpiEconMFG.MFGEpiEcon(T_End = 2.0, ξ = 0.001)
result = run(p=p)
EpiEconMFG.save_all_figures(result,p; outdir=joinpath(@__DIR__,"..","outputs/quasi-static-HJB_T2_xi0001"))

