include("L_loadAll.jl")


"""
	run()

Convenience entry point for quick experimentation.

Builds default parameters `p = MFGEpiEcon()`, creates a simple test distribution,
and runs `solveModel` with `show_progress=true`.
"""
function run()
    p = MFGEpiEcon()
    F0 = create_test_distribution(p)
    result = solveModel(p, F0; show_progress=true)

    return result
end