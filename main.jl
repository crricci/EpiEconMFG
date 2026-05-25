include("src/EpiEconMFG.jl")


"""
	run(; p=EpiEconMFG.MFGEpiEcon(), F0=EpiEconMFG.create_test_distribution(p), show_progress=true, kwargs...)

Convenience entry point for quick experimentation.

Builds default parameters `p = MFGEpiEcon()`, creates a simple test distribution,
and runs `solveModel` with `show_progress=true`.
"""
function run(;
    p = EpiEconMFG.MFGEpiEcon(),
    F0 = EpiEconMFG.create_test_distribution(p),
    show_progress = true,
    kwargs...,
)
    result = EpiEconMFG.solveModel(p, F0; show_progress = show_progress, kwargs...)

    return result
end
