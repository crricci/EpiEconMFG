using CairoMakie

include(joinpath(@__DIR__, "..", "main.jl"))

const BASELINE_CASE = (
    dir = "dynamic_BASELINE",
    label = "Baseline",
    color = :black,
)

const COMP_CASES = [
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
        label = "Linear \u03BE(k): 0 to 0.25",
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

function relative_C_at_time(result, p, idx)
    Ft = result.F[idx]
    den = Ft.ϕSt .+ Ft.ϕIt .+ Ft.ϕCt .+ Ft.ϕRt
    rel = similar(den, Float64)
    @inbounds for i in eachindex(den)
        d = Float64(den[i])
        rel[i] = (isfinite(d) && d > 0) ? Float64(Ft.ϕCt[i]) / d : 0.0
    end
    return rel
end

function first_peak_of_integrated_C(result, p)
    Nt = length(result.t)
    Nt == length(result.F) || error("Inconsistent result lengths")

    Ctot = Vector{Float64}(undef, Nt)
    @inbounds for n in 1:Nt
        Ctot[n] = sum(result.F[n].ϕCt) * p.Δk
    end
    peak_val = maximum(Ctot)
    peak_idx = findfirst(==(peak_val), Ctot)
    peak_idx === nothing && error("Cannot locate peak of integrated C")
    return peak_idx, peak_val, Ctot
end

function load_case(case)
    outdir = joinpath(@__DIR__, "..", "outputs", case.dir)
    meta = read_metadata(outdir)
    p = params_from_metadata(meta)
    result = EpiEconMFG.load_solution_csv(outdir, p)
    peak_idx, peak_val, Ctot = first_peak_of_integrated_C(result, p)
    relC = relative_C_at_time(result, p, peak_idx)
    return (
        t = result.t[peak_idx],
        k = collect(p.k),
        relC = relC,
        peak_idx = peak_idx,
        peak_val = peak_val,
        Ctot = Ctot,
        converged = get(meta, "converged", ""),
        iterations = get(meta, "iterations", ""),
    )
end

function build_ratio_data(baseline, case)
    if length(baseline.k) != length(case.k)
        error("Grid length mismatch between baseline and $(case)")
    end
    if maximum(abs.(baseline.k .- case.k)) > 1e-10
        error("Grid mismatch between baseline and $(case)")
    end

    epsy = eps(Float64)
    k = baseline.k[2:end]
    base_rel = max.(baseline.relC[2:end], epsy)
    case_rel = max.(case.relC[2:end], epsy)
    ratio = case_rel ./ base_rel
    return (k = k, ratio = ratio)
end

function plot_ratio_comparison(; logscale::Bool, outdir, filename)
    mkpath(outdir)

    baseline = load_case(BASELINE_CASE)
    cases = [load_case(case) for case in COMP_CASES]
    case_t = Dict(case.dir => data.t for (case, data) in zip(COMP_CASES, cases))
    plotted = [build_ratio_data(baseline, case) for case in cases]

    rmin = minimum(minimum(data.ratio) for data in plotted)
    rmax = maximum(maximum(data.ratio) for data in plotted)
    epsy = eps(Float64)

    fig = CairoMakie.Figure(size = (980, 560))
    ax = if logscale
        CairoMakie.Axis(
            fig[1, 1],
            title = "Relative contained share at the first peak, normalized by baseline",
            xlabel = "k",
            ylabel = "(C/(S+I+C+R)) / baseline",
            yscale = log10,
        )
    else
        CairoMakie.Axis(
            fig[1, 1],
            title = "Relative contained share at the first peak, normalized by baseline",
            xlabel = "k",
            ylabel = "(C/(S+I+C+R)) / baseline",
        )
    end

    handles = Any[]
    labels = String[]
    for (case, data) in zip(COMP_CASES, plotted)
        label = "$(case.label) (t*=$(round(case_t[case.dir], digits = 2)))"
        line = CairoMakie.lines!(ax, data.k, logscale ? max.(data.ratio, epsy) : data.ratio; color = case.color, linewidth = 2.7)
        push!(handles, line)
        push!(labels, label)
    end

    CairoMakie.axislegend(
        ax,
        handles,
        labels;
        position = :rt,
        framevisible = true,
        backgroundcolor = (:white, 1.0),
    )

    if logscale
        CairoMakie.ylims!(ax, max(epsy, rmin / 1.5), rmax * 1.5)
    else
        pad = max(0.05 * (rmax - rmin), 1e-6)
        CairoMakie.ylims!(ax, max(0.0, rmin - pad), rmax + pad)
    end

    outpath = joinpath(outdir, filename)
    CairoMakie.save(outpath, fig)
    return outpath
end

outdir = joinpath(@__DIR__, "..", "outputs", "peak_C_ratio_comparison")
path1 = plot_ratio_comparison(; logscale = false, outdir = outdir, filename = "figure_peak_C_ratio_baseline_linear.pdf")
path2 = plot_ratio_comparison(; logscale = true, outdir = outdir, filename = "figure_peak_C_ratio_baseline_log.pdf")
println("Saved $path1")
println("Saved $path2")
