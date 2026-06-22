using CairoMakie

include(joinpath(@__DIR__, "..", "main.jl"))

const CASES = [
    (
        dir = "dynamic_BASELINE",
        label = "Baseline",
        color = :black,
    ),
    (
        dir = "dynamic_VACCINEFREE",
        label = "Free vaccine",
        color = :dodgerblue3,
    ),
    (
        dir = "dynamic_NOVACCINE",
        label = "No vaccine",
        color = :firebrick3,
    ),
    (
        dir = "dynamic_VACCINE_LINEAR_K_0_025",
        label = "Linear ξ(k): 0 to 0.25",
        color = :darkgreen,
    ),
]

function read_metadata(outdir)
    path = joinpath(outdir, "metadata.csv")
    isfile(path) || error("Missing metadata.csv in $outdir")

    meta = Dict{String,String}()
    for (i, line) in enumerate(eachline(path))
        i == 1 && continue
        isempty(strip(line)) && continue
        parts = split(line, ","; limit = 2)
        length(parts) == 2 || error("Malformed metadata line in $path: $line")
        meta[parts[1]] = parts[2]
    end
    return meta
end

function params_from_metadata(meta)
    return EpiEconMFG.MFGEpiEcon(
        MaxK = parse(Float64, meta["MaxK"]),
        Δk = parse(Float64, meta["DeltaK"]),
        T_End = parse(Float64, meta["T_End"]),
    )
end

function load_gini_case(case)
    outdir = joinpath(@__DIR__, "..", "outputs", case.dir)
    meta = read_metadata(outdir)
    p = params_from_metadata(meta)
    result = EpiEconMFG.load_solution_csv(outdir, p)
    return (
        t = result.t,
        gini = EpiEconMFG.wealth_gini_over_time(result, p),
        converged = get(meta, "converged", ""),
        iterations = get(meta, "iterations", ""),
    )
end

function plot_gini_comparison(;
    outdir = joinpath(@__DIR__, "..", "outputs", "gini_comparison"),
    filename = "figure_gini_wealth_four_cases.pdf",
)
    mkpath(outdir)

    loaded = [load_gini_case(case) for case in CASES]
    t_ref = loaded[1].t
    for (case, data) in zip(CASES, loaded)
        length(data.t) == length(t_ref) || error("Time-grid length mismatch for $(case.dir)")
        maximum(abs.(data.t .- t_ref)) <= 1e-10 || error("Time-grid mismatch for $(case.dir)")
    end

    gmin = minimum(minimum(data.gini) for data in loaded)
    gmax = maximum(maximum(data.gini) for data in loaded)
    pad = max(0.05 * (gmax - gmin), 1e-6)

    fig = CairoMakie.Figure(size = (980, 560))
    ax = CairoMakie.Axis(
        fig[1, 1],
        title = "Wealth Gini index across vaccine policies",
        xlabel = "t",
        ylabel = "Gini",
    )

    handles = Any[]
    labels = String[]
    for (case, data) in zip(CASES, loaded)
        line = CairoMakie.lines!(ax, data.t, data.gini; color = case.color, linewidth = 2.7)
        push!(handles, line)
        push!(labels, case.label)
    end
    CairoMakie.axislegend(
        ax,
        handles,
        labels;
        position = :rt,
        framevisible = true,
        backgroundcolor = (:white, 1.0),
    )
    CairoMakie.ylims!(ax, gmin - pad, gmax + pad)

    outpath = joinpath(outdir, filename)
    CairoMakie.save(outpath, fig)
    return outpath
end

outpath = plot_gini_comparison()
println("Saved $outpath")
