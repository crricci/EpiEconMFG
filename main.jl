include("src/EpiEconMFG.jl")

"""
	run(; p=EpiEconMFG.MFGEpiEcon(), F0=EpiEconMFG.initial_distribution(p), show_progress=true, outdir=nothing, kwargs...)

Convenience entry point for quick experimentation.

Builds default parameters `p = MFGEpiEcon()`, creates the configured initial distribution,
and runs `solveModel` with `show_progress=true`. If `outdir` is provided, the
solution is also exported to CSV files in that directory.
"""
function run(;
    p = EpiEconMFG.MFGEpiEcon(),
    F0 = EpiEconMFG.initial_distribution(p),
    show_progress = true,
    outdir = nothing,
    kwargs...,
)
    result = EpiEconMFG.solveModel(p, F0; show_progress = show_progress, kwargs...)
    if outdir !== nothing
        EpiEconMFG.save_solution_csv(result, p; outdir = outdir)
    end

    return result
end

"""
    run_dynamic(; p=EpiEconMFG.MFGEpiEcon(), F0=EpiEconMFG.initial_distribution(p), show_progress=true, outdir=nothing, kwargs...)

Convenience entry point for the fully dynamic forward-backward solver.
If `outdir` is provided, the solution is also exported to CSV files in that
directory.
"""
function run_dynamic(;
    p = EpiEconMFG.MFGEpiEcon(),
    F0 = EpiEconMFG.initial_distribution(p),
    show_progress = true,
    outdir = nothing,
    kwargs...,
)
    result = EpiEconMFG.solveModelDynamic(p, F0; show_progress = show_progress, kwargs...)
    if outdir !== nothing
        EpiEconMFG.save_solution_csv(result, p; outdir = outdir)
    end

    return result
end
