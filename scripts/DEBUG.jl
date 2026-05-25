using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))
include(joinpath(@__DIR__, "..", "main.jl"))


p = EpiEconMFG.MFGEpiEcon(ξ = 0.1)
result = run(p=p)
EpiEconMFG.save_all_figures(result,p; outdir=joinpath(@__DIR__,"..","outputs/vacc_cost_0.1"))

p = EpiEconMFG.MFGEpiEcon(ξ = 0.01)
result = run(p=p)
EpiEconMFG.save_all_figures(result,p; outdir=joinpath(@__DIR__,"..","outputs/vacc_cost_0.01"))

p = EpiEconMFG.MFGEpiEcon(ξ = 0.001)
result = run(p=p)
EpiEconMFG.save_all_figures(result,p; outdir=joinpath(@__DIR__,"..","outputs/vacc_cost_0.001"))