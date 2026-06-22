using Pkg
Pkg.activate(joinpath(@__DIR__, ".."))

include(joinpath(@__DIR__, "..", "main.jl"))

const CASES = [
    (
        name = "dynamic_NoEpidemic_long_restart_from_finalS",
        source_dir = joinpath(@__DIR__, "..", "outputs", "dynamic_NoEpidemic_long"),
        outdir = joinpath(@__DIR__, "..", "outputs", "dynamic_NoEpidemic_long_restart_from_finalS"),
    ),
    (
        name = "dynamic_NoEpidemic_long_narrow_restart_from_finalS",
        source_dir = joinpath(@__DIR__, "..", "outputs", "dynamic_NoEpidemic_long_narrow"),
        outdir = joinpath(@__DIR__, "..", "outputs", "dynamic_NoEpidemic_long_narrow_restart_from_finalS"),
    ),
]

function read_metadata(source_dir)
    path = joinpath(source_dir, "metadata.csv")
    if !isfile(path)
        error("Missing metadata file: $path")
    end

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

function final_susceptible_from_csv(source_dir, p)
    path = joinpath(source_dir, "distributions.csv")
    if !isfile(path)
        error("Missing distributions file: $path")
    end

    S = zeros(Float64, p.Nk)
    I = zeros(Float64, p.Nk)
    C = zeros(Float64, p.Nk)
    R = zeros(Float64, p.Nk)
    current_time_index = 0
    current_time = NaN

    for (line_no, line) in enumerate(eachline(path))
        line_no == 1 && continue
        isempty(strip(line)) && continue

        parts = split(line, ",")
        length(parts) >= 8 || error("Malformed distributions line $line_no in $path")
        time_index = parse(Int, parts[1])
        t = parse(Float64, parts[2])
        k_index = parse(Int, parts[3])

        if time_index != current_time_index
            fill!(S, 0.0)
            fill!(I, 0.0)
            fill!(C, 0.0)
            fill!(R, 0.0)
            current_time_index = time_index
            current_time = t
        end

        if !(1 <= k_index <= p.Nk)
            error("k_index=$k_index outside 1:$(p.Nk) in $path line $line_no")
        end
        S[k_index] = parse(Float64, parts[5])
        I[k_index] = parse(Float64, parts[6])
        C[k_index] = parse(Float64, parts[7])
        R[k_index] = parse(Float64, parts[8])
    end

    total_mass = (sum(S) + sum(I) + sum(C) + sum(R)) * p.Δk
    susceptible_mass = sum(S) * p.Δk
    other_mass = (sum(I) + sum(C) + sum(R)) * p.Δk
    if !(isfinite(susceptible_mass) && susceptible_mass > 0)
        error("Final susceptible mass in $source_dir is non-positive: $susceptible_mass")
    end

    S0 = max.(S, 0.0)
    S0 ./= sum(S0) * p.Δk

    return (
        F0 = (
            ϕSt = S0,
            ϕIt = zeros(Float64, p.Nk),
            ϕCt = zeros(Float64, p.Nk),
            ϕRt = zeros(Float64, p.Nk),
        ),
        source_time_index = current_time_index,
        source_time = current_time,
        source_total_mass = total_mass,
        source_susceptible_mass = susceptible_mass,
        source_other_mass = other_mass,
    )
end

function write_restart_summary(outdir, case, info, result, p)
    open(joinpath(outdir, "restart_summary.csv"), "w") do io
        println(io, "key,value")
        println(io, "case,$(case.name)")
        println(io, "source_dir,$(case.source_dir)")
        println(io, "source_time_index,$(info.source_time_index)")
        println(io, "source_time,$(info.source_time)")
        println(io, "source_total_mass,$(info.source_total_mass)")
        println(io, "source_susceptible_mass,$(info.source_susceptible_mass)")
        println(io, "source_other_mass,$(info.source_other_mass)")
        println(io, "new_initial_mass,$((sum(info.F0.ϕSt) + sum(info.F0.ϕIt) + sum(info.F0.ϕCt) + sum(info.F0.ϕRt)) * p.Δk)")
        println(io, "converged,$(result.converged)")
        println(io, "iterations,$(result.iterations)")
        println(io, "final_err,$(isempty(result.diagnostics.err) ? NaN : last(result.diagnostics.err))")
    end
end

function copy_distribution_local(Ft)
    return (ϕSt = copy(Ft.ϕSt), ϕIt = copy(Ft.ϕIt), ϕCt = copy(Ft.ϕCt), ϕRt = copy(Ft.ϕRt))
end

function copy_value_local(V)
    return (VS = copy(V.VS), VI = copy(V.VI), VC = copy(V.VC), VR = copy(V.VR))
end

function seed_paths_from_source(source_result, F0, t_full)
    Vseed = copy_value_local(source_result.V[end])
    Cseed = deepcopy(source_result.controls[end])
    F_path = [copy_distribution_local(F0) for _ in eachindex(t_full)]
    V_path = [copy_value_local(Vseed) for _ in eachindex(t_full)]
    controls_path = [deepcopy(Cseed) for _ in eachindex(t_full)]
    return F_path, V_path, controls_path
end

function solve_dynamic_from_seed_paths(p, F0, t_full, F_path_old, V_path_old, controls_path_old; show_progress = true)
    dt_eff = p.T_End / (length(t_full) - 1)
    VT = copy_value_local(V_path_old[end])

    err_hist = Float64[]
    errF_hist = Float64[]
    errV_hist = Float64[]
    errW_hist = Float64[]
    errR_hist = Float64[]
    mass_error_hist = Float64[]
    min_density_hist = Float64[]
    max_negative_hist = Float64[]
    norm_correction_hist = Float64[]
    hjb_diagnostics_hist = Any[]
    fp_diagnostics_hist = Any[]

    converged = false
    iterations = 0
    prices_old = EpiEconMFG.compute_prices_path(F_path_old, controls_path_old, p)

    for it in 1:p.maxIterDynamic
        iterations = it
        F_before = F_path_old
        V_before = V_path_old
        prices_old = EpiEconMFG.compute_prices_path(F_before, controls_path_old, p)

        if show_progress
            println("seeded dynamic Picard $it/$(p.maxIterDynamic): backward HJB")
            flush(stdout)
        end

        V_path_new, hjb_diag = EpiEconMFG.solve_hjb_backward(
            VT,
            F_before,
            controls_path_old,
            prices_old.w,
            prices_old.r,
            prices_old.LI,
            p;
            V_guess_path = V_before,
            dt = dt_eff,
            t = t_full,
            show_progress = false,
        )

        controls_path_new = EpiEconMFG.compute_dynamic_controls_path(V_path_new, F_before, prices_old, p; t_path = t_full)

        if show_progress
            println("seeded dynamic Picard $it/$(p.maxIterDynamic): forward FP")
            flush(stdout)
        end

        F_path_new, fp_diag = EpiEconMFG.solve_fp_forward_dynamic(
            F0,
            controls_path_new,
            prices_old.w,
            prices_old.r,
            prices_old.LI,
            p;
            dt = dt_eff,
            t = t_full,
        )

        prices_new = EpiEconMFG.compute_prices_path(F_path_new, controls_path_new, p)

        errF = EpiEconMFG.max_distribution_distance(F_path_new, F_before)
        errV = EpiEconMFG.max_value_path_distance(V_path_new, V_before)
        errW = maximum(abs.(prices_new.w .- prices_old.w))
        errR = maximum(abs.(prices_new.r .- prices_old.r))
        err = maximum((errF, errV, errW, errR))
        mass_error = EpiEconMFG.path_mass_error(F_path_new, p)
        min_density = EpiEconMFG.path_min_density(F_path_new)
        max_negative = minimum(fp_diag.max_negative_before_project)
        max_norm_correction = maximum(fp_diag.normalization_correction)

        push!(err_hist, err)
        push!(errF_hist, errF)
        push!(errV_hist, errV)
        push!(errW_hist, errW)
        push!(errR_hist, errR)
        push!(mass_error_hist, mass_error)
        push!(min_density_hist, min_density)
        push!(max_negative_hist, max_negative)
        push!(norm_correction_hist, max_norm_correction)
        push!(hjb_diagnostics_hist, hjb_diag)
        push!(fp_diagnostics_hist, fp_diag)

        if show_progress
            println(
                "seeded dynamic Picard $it: errF=$(round(errF, sigdigits=4)) " *
                "errV=$(round(errV, sigdigits=4)) errW=$(round(errW, sigdigits=4)) " *
                "err=$(round(err, sigdigits=4)) mass_err=$(round(mass_error, sigdigits=4)) " *
                "minF=$(round(min_density, sigdigits=4))"
            )
            flush(stdout)
        end

        if err < p.tolDynamic
            converged = true
            F_path_old = F_path_new
            V_path_old = V_path_new
            controls_path_old = controls_path_new
            prices_old = prices_new
            break
        end

        F_damped = EpiEconMFG.damp_distribution_path(F_before, F_path_new, p.ωF_dynamic)
        V_damped = EpiEconMFG.damp_value_path(V_before, V_path_new, p.ωV_dynamic; terminal = VT)
        prices_for_damped = EpiEconMFG.compute_prices_path(F_damped, controls_path_new, p)
        controls_damped = EpiEconMFG.compute_dynamic_controls_path(V_damped, F_damped, prices_for_damped, p; t_path = t_full)

        F_path_old = F_damped
        V_path_old = V_damped
        controls_path_old = controls_damped
        prices_old = EpiEconMFG.compute_prices_path(F_path_old, controls_path_old, p)
    end

    V_saved = [
        (
            VS = copy(V_path_old[i].VS),
            VI = copy(V_path_old[i].VI),
            VC = copy(V_path_old[i].VC),
            VR = copy(V_path_old[i].VR),
            w = prices_old.w[i],
            r = prices_old.r[i],
            LI = prices_old.LI[i],
        )
        for i in eachindex(t_full)
    ]

    diagnostics = (
        err = err_hist,
        errF = errF_hist,
        errV = errV_hist,
        errW = errW_hist,
        errR = errR_hist,
        mass_error = mass_error_hist,
        min_density = min_density_hist,
        max_negative_before_projection = max_negative_hist,
        normalization_correction = norm_correction_hist,
        hjb = hjb_diagnostics_hist,
        fp = fp_diagnostics_hist,
    )

    return (
        t = t_full,
        F = F_path_old,
        V = V_saved,
        controls = controls_path_old,
        prices = (w = prices_old.w, r = prices_old.r, K = prices_old.K, L = prices_old.L, LI = prices_old.LI),
        aggregates = (K = prices_old.K, L = prices_old.L, LI = prices_old.LI),
        diagnostics = diagnostics,
        converged = converged,
        iterations = iterations,
        method = :forward_backward_dynamic_seeded_from_csv,
    )
end

function build_restart_parameters(meta)
    MaxK = parse(Float64, meta["MaxK"])
    Δk = parse(Float64, meta["DeltaK"])
    ξ = haskey(meta, "xi") ? parse(Float64, meta["xi"]) : EpiEconMFG.MFGEpiEcon().ξ

    return EpiEconMFG.MFGEpiEcon(
        I0 = 0.0,
        T_End = 60.0,
        Δk = Δk,
        MaxK = MaxK,
        ξ = ξ,
        maxIterDynamic = 1000000000,
        maxitHJBvalue = Int(1e6),
        maxitWage = 100000000,
        truncateKPlots = false,
        dynamicVerbose = false,
        verbose = false,
    )
end

function selected_cases()
    isempty(ARGS) && return CASES
    wanted = Set(ARGS)
    cases = [case for case in CASES if case.name in wanted || basename(case.source_dir) in wanted]
    isempty(cases) && error("No matching cases. Available: $(join((case.name for case in CASES), ", "))")
    return cases
end

for case in selected_cases()
    println("=== $(case.name) ===")
    meta = read_metadata(case.source_dir)
    p = build_restart_parameters(meta)
    info = final_susceptible_from_csv(case.source_dir, p)

    mkpath(case.outdir)
    open(joinpath(case.outdir, "output.log"), "w") do logio
        redirect_stdout(logio) do
            redirect_stderr(logio) do
                println("source_dir=$(case.source_dir)")
                println("source_time=$(info.source_time)")
                println("source_total_mass=$(info.source_total_mass)")
                println("source_susceptible_mass=$(info.source_susceptible_mass)")
                println("source_other_mass=$(info.source_other_mass)")
                flush(stdout)

                source_result = EpiEconMFG.load_solution_csv(case.source_dir, p)
                Nstep_eff = Int(ceil(p.T_End / p.Δt))
                t_full = collect(range(0.0, p.T_End; length = Nstep_eff + 1))
                F_seed, V_seed, controls_seed = seed_paths_from_source(source_result, info.F0, t_full)
                result = solve_dynamic_from_seed_paths(p, info.F0, t_full, F_seed, V_seed, controls_seed; show_progress = true)
                EpiEconMFG.save_solution_csv(result, p; outdir = case.outdir)
                write_restart_summary(case.outdir, case, info, result, p)
                EpiEconMFG.save_all_figures(result, p; outdir = case.outdir, progress = true)

                println(
                    "DONE $(case.name): converged=$(result.converged) " *
                    "iterations=$(result.iterations) " *
                    "final_err=$(isempty(result.diagnostics.err) ? NaN : last(result.diagnostics.err))"
                )
                flush(stdout)
            end
        end
    end

    println("finished $(case.name); outputs in $(case.outdir)")
end
