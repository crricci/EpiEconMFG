function copy_distribution(Ft)
    return (ϕSt = copy(Ft.ϕSt), ϕIt = copy(Ft.ϕIt), ϕCt = copy(Ft.ϕCt), ϕRt = copy(Ft.ϕRt))
end

function simple_value(V)
    return (VS = copy(V.VS), VI = copy(V.VI), VC = copy(V.VC), VR = copy(V.VR))
end

function max_distribution_distance(A_path, B_path)
    err = 0.0
    @inbounds for n in eachindex(A_path)
        A = A_path[n]
        B = B_path[n]
        err = max(err, maximum(abs.(A.ϕSt .- B.ϕSt)))
        err = max(err, maximum(abs.(A.ϕIt .- B.ϕIt)))
        err = max(err, maximum(abs.(A.ϕCt .- B.ϕCt)))
        err = max(err, maximum(abs.(A.ϕRt .- B.ϕRt)))
    end
    return err
end

function max_value_path_distance(A_path, B_path)
    err = 0.0
    @inbounds for n in eachindex(A_path)
        err = max(err, max_value_distance(A_path[n], B_path[n]))
    end
    return err
end

function damp_distribution_path(old_path, new_path, ω)
    Nt = length(old_path)
    out = Vector{Any}(undef, Nt)
    @inbounds for n in 1:Nt
        old = old_path[n]
        new = new_path[n]
        out[n] = (
            ϕSt = (1 - ω) .* old.ϕSt .+ ω .* new.ϕSt,
            ϕIt = (1 - ω) .* old.ϕIt .+ ω .* new.ϕIt,
            ϕCt = (1 - ω) .* old.ϕCt .+ ω .* new.ϕCt,
            ϕRt = (1 - ω) .* old.ϕRt .+ ω .* new.ϕRt,
        )
    end
    return out
end

function damp_value_path(old_path, new_path, ω; terminal=nothing)
    Nt = length(old_path)
    out = Vector{Any}(undef, Nt)
    @inbounds for n in 1:Nt
        old = old_path[n]
        new = new_path[n]
        out[n] = (
            VS = (1 - ω) .* old.VS .+ ω .* new.VS,
            VI = (1 - ω) .* old.VI .+ ω .* new.VI,
            VC = (1 - ω) .* old.VC .+ ω .* new.VC,
            VR = (1 - ω) .* old.VR .+ ω .* new.VR,
        )
    end
    if terminal !== nothing
        out[end] = copy_value(terminal)
    end
    return out
end

function path_mass_error(F_path, p)
    err = 0.0
    @inbounds for Ft in F_path
        err = max(err, abs(distribution_mass(Ft, p) - 1.0))
    end
    return err
end

function path_min_density(F_path)
    m = Inf
    @inbounds for Ft in F_path
        m = min(m, minimum(Ft.ϕSt), minimum(Ft.ϕIt), minimum(Ft.ϕCt), minimum(Ft.ϕRt))
    end
    return m
end

function saved_indices(Nt, save_stride)
    if save_stride < 1
        error("save_stride must be >= 1")
    end
    idx = collect(1:save_stride:Nt)
    if last(idx) != Nt
        push!(idx, Nt)
    end
    return idx
end

function compute_dynamic_controls_path(V_path, F_path, prices, p; t_path=nothing)
    Nt = length(V_path)
    if length(F_path) != Nt
        error("V_path and F_path must have the same length")
    end
    controls = Vector{Any}(undef, Nt)
    dcache = DerivDkCache(eltype(V_path[1].VS), p.Nk)
    tnodes = isnothing(t_path) ? collect(range(0, p.T_End, length=Nt)) : t_path

    @inbounds for n in 1:Nt
        controls[n] = compute_time_dependent_policies(
            V_path[n],
            F_path[n],
            p;
            w = prices.w[n],
            r = prices.r[n],
            LI = prices.LI[n],
            t = tnodes[n],
            deriv_cache = dcache,
        )
    end

    return controls
end

function constant_initial_dynamic_guess(p, F0, t_full)
    Nt = length(t_full)
    F_path = [copy_distribution(F0) for _ in 1:Nt]
    V_path = [
        (VS = zeros(Float64, p.Nk), VI = zeros(Float64, p.Nk), VC = zeros(Float64, p.Nk), VR = zeros(Float64, p.Nk))
        for _ in 1:Nt
    ]

    l0 = (
        lS = ones(Float64, p.Nk),
        lI = ones(Float64, p.Nk),
        lC = zeros(Float64, p.Nk),
        lR = ones(Float64, p.Nk),
    )
    controls0 = [(
        lOpt = l0,
        q_rate = zeros(Float64, p.Nk),
        bS = zeros(Float64, p.Nk),
        bI = zeros(Float64, p.Nk),
        bC = zeros(Float64, p.Nk),
        bR = zeros(Float64, p.Nk),
    ) for _ in 1:Nt]
    prices = compute_prices_path(F_path, controls0, p)
    controls = compute_dynamic_controls_path(V_path, F_path, prices, p; t_path=t_full)
    return F_path, V_path, controls
end

function quasistatic_initial_dynamic_guess(p, F0; show_progress=false)
    result = solveModel(p, F0; show_progress=show_progress, save_stride=1, HJB_every=1)
    F_path = [copy_distribution(Ft) for Ft in result.F]
    V_path = [simple_value(Vt) for Vt in result.V]
    controls_path = deepcopy(result.controls)
    return result.t, F_path, V_path, controls_path
end

function dynamic_initial_guess(p, F0, t_full; show_progress=false, initial_guess=p.dynamicInitialGuess)
    if initial_guess == :quasistatic
        t_guess, F_path, V_path, controls_path = quasistatic_initial_dynamic_guess(p, F0; show_progress=show_progress)
        if length(t_guess) != length(t_full) || maximum(abs.(t_guess .- t_full)) > sqrt(eps(Float64))
            error("Quasi-static initial guess returned a different time grid")
        end
        return F_path, V_path, controls_path
    elseif initial_guess == :constant || initial_guess == :zero
        return constant_initial_dynamic_guess(p, F0, t_full)
    else
        error("Unsupported dynamicInitialGuess=$(initial_guess). Supported values: :quasistatic, :constant, :zero")
    end
end

"""
    solveModelDynamic(p, F0; show_progress=true, save_stride=1, kwargs...)

Solve the fully dynamic forward-backward MFG system.

The solver uses a Picard iteration on the full distribution path. At each outer
iteration it computes aggregate price paths from the current path, solves the HJB
backward in time with backward Euler, then advances the FP/KFE forward in time.
"""
function solveModelDynamic(
    p,
    F0;
    show_progress = p.dynamicVerbose,
    save_stride::Int = 1,
    initial_guess = p.dynamicInitialGuess,
    terminal = p.dynamicTerminal,
    quasistatic_show_progress = false,
)
    if save_stride < 1
        error("save_stride must be >= 1")
    end
    if terminal != :fixed_quasistatic
        error("Unsupported dynamicTerminal=$terminal. Supported value: :fixed_quasistatic")
    end

    if !(isfinite(p.T_End) && p.T_End > 0)
        error("p.T_End must be finite and > 0")
    end
    if !(isfinite(p.Δt) && p.Δt > 0)
        error("p.Δt must be finite and > 0")
    end

    Nstep_eff = Int(ceil(p.T_End / p.Δt))
    t_full = collect(range(0, p.T_End, length=Nstep_eff + 1))
    dt_eff = p.T_End / Nstep_eff

    F_path_old, V_path_old, controls_path_old = dynamic_initial_guess(
        p,
        F0,
        t_full;
        show_progress = quasistatic_show_progress,
        initial_guess = initial_guess,
    )

    VT = copy_value(V_path_old[end])

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
    prices_old = compute_prices_path(F_path_old, controls_path_old, p)

    for it in 1:p.maxIterDynamic
        iterations = it
        F_before = F_path_old
        V_before = V_path_old
        prices_old = compute_prices_path(F_before, controls_path_old, p)

        if show_progress
            println("dynamic Picard $it/$(p.maxIterDynamic): backward HJB")
        end

        V_path_new, hjb_diag = solve_hjb_backward(
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

        controls_path_new = compute_dynamic_controls_path(V_path_new, F_before, prices_old, p; t_path=t_full)

        if show_progress
            println("dynamic Picard $it/$(p.maxIterDynamic): forward FP")
        end

        F_path_new, fp_diag = solve_fp_forward_dynamic(
            F0,
            controls_path_new,
            prices_old.w,
            prices_old.r,
            prices_old.LI,
            p;
            dt = dt_eff,
            t = t_full,
        )

        prices_new = compute_prices_path(F_path_new, controls_path_new, p)

        errF = max_distribution_distance(F_path_new, F_before)
        errV = max_value_path_distance(V_path_new, V_before)
        errW = maximum(abs.(prices_new.w .- prices_old.w))
        errR = maximum(abs.(prices_new.r .- prices_old.r))
        mass_error = path_mass_error(F_path_new, p)
        min_density = path_min_density(F_path_new)
        max_negative = minimum(fp_diag.max_negative_before_project)
        max_norm_correction = maximum(fp_diag.normalization_correction)

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
                "dynamic Picard $it: errF=$(round(errF, sigdigits=4)) " *
                "errV=$(round(errV, sigdigits=4)) errW=$(round(errW, sigdigits=4)) " *
                "mass_err=$(round(mass_error, sigdigits=4)) minF=$(round(min_density, sigdigits=4))"
            )
        end

        if errF < p.tolDynamic
            converged = true
            F_path_old = F_path_new
            V_path_old = V_path_new
            controls_path_old = controls_path_new
            prices_old = prices_new
            break
        end

        F_damped = damp_distribution_path(F_before, F_path_new, p.ωF_dynamic)
        V_damped = damp_value_path(V_before, V_path_new, p.ωV_dynamic; terminal=VT)

        prices_for_damped = compute_prices_path(F_damped, controls_path_new, p)
        controls_damped = compute_dynamic_controls_path(V_damped, F_damped, prices_for_damped, p; t_path=t_full)

        F_path_old = F_damped
        V_path_old = V_damped
        controls_path_old = controls_damped
        prices_old = compute_prices_path(F_path_old, controls_path_old, p)
    end

    idx = saved_indices(length(t_full), save_stride)
    t_saved = t_full[idx]
    F_saved = F_path_old[idx]
    controls_saved = controls_path_old[idx]
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
        for i in idx
    ]

    diagnostics = (
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

    aggregates = (
        K = prices_old.K[idx],
        L = prices_old.L[idx],
        LI = prices_old.LI[idx],
    )
    prices = (
        w = prices_old.w[idx],
        r = prices_old.r[idx],
        K = prices_old.K[idx],
        L = prices_old.L[idx],
        LI = prices_old.LI[idx],
    )

    return (
        t = t_saved,
        F = F_saved,
        V = V_saved,
        controls = controls_saved,
        prices = prices,
        aggregates = aggregates,
        diagnostics = diagnostics,
        converged = converged,
        iterations = iterations,
        method = :forward_backward_dynamic,
    )
end
