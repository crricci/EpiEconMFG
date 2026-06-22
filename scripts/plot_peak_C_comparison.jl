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

function plot_peak_C_comparison(;
    outdir = joinpath(@__DIR__, "..", "outputs", "peak_C_comparison"),
    filename = "figure_peak_C_share_four_cases.pdf",
)
    mkpath(outdir)

    loaded = [load_case(case) for case in CASES]
    epsy = eps(Float64)
    plotted = [
        (
            k = data.k[2:end],
            relC = max.(data.relC[2:end], epsy),
        )
        for data in loaded
    ]

    gmin = minimum(minimum(data.relC) for data in plotted)
    gmax = maximum(maximum(data.relC) for data in plotted)

    fig = CairoMakie.Figure(size = (980, 560))
    ax = CairoMakie.Axis(
        fig[1, 1],
        title = "Relative contained share at the first peak of integrated C",
        xlabel = "k",
        ylabel = "C/(S+I+C+R)",
        yscale = log10,
    )

    handles = Any[]
    labels = String[]
    for (case, data, pdata) in zip(CASES, loaded, plotted)
        label = "$(case.label) (t*=$(round(data.t, digits = 2)))"
        line = CairoMakie.lines!(ax, pdata.k, pdata.relC; color = case.color, linewidth = 2.7)
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
    CairoMakie.ylims!(ax, max(epsy, gmin / 1.5), min(1.0, gmax * 1.5))

    outpath = joinpath(outdir, filename)
    CairoMakie.save(outpath, fig)
    return outpath
end

outpath = plot_peak_C_comparison()
println("Saved $outpath")
