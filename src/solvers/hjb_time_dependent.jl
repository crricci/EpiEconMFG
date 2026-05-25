function stack_value(V)
    return vcat(V.VS, V.VI, V.VC, V.VR)
end

function unstack_value(v, p)
    Nk = p.Nk
    return (
        VS = v[1:Nk],
        VI = v[Nk + 1:2Nk],
        VC = v[2Nk + 1:3Nk],
        VR = v[3Nk + 1:4Nk],
    )
end

function copy_value(V)
    return (VS = copy(V.VS), VI = copy(V.VI), VC = copy(V.VC), VR = copy(V.VR))
end

function max_value_distance(A, B)
    return maximum((
        maximum(abs.(A.VS .- B.VS)),
        maximum(abs.(A.VI .- B.VI)),
        maximum(abs.(A.VC .- B.VC)),
        maximum(abs.(A.VR .- B.VR)),
    ))
end

function compute_HJB_dynamic_policies(Vn, Fn, w, r, LI, t, p; deriv_cache=nothing)
    return compute_time_dependent_policies(Vn, Fn, p; w=w, r=r, LI=LI, t=t, deriv_cache=deriv_cache)
end

function build_HJB_dynamic_step_system(
    Vn_guess,
    Vnext,
    Fn,
    controls_old_n,
    w,
    r,
    LI,
    t,
    dt,
    p;
    deriv_cache=nothing,
)
    pol = compute_HJB_dynamic_policies(Vn_guess, Fn, w, r, LI, t, p; deriv_cache=deriv_cache)

    Nk = p.Nk
    nstate = 4 * Nk
    Δk = p.Δk

    I = Int[]
    J = Int[]
    X = eltype(Vn_guess.VS)[]
    rhs = zeros(eltype(Vn_guess.VS), nstate)

    sizehint!(I, 9nstate)
    sizehint!(J, 9nstate)
    sizehint!(X, 9nstate)

    idx(state, i) = (state - 1) * Nk + i

    function push_entry!(row, col, val)
        push!(I, row)
        push!(J, col)
        push!(X, val)
        return nothing
    end

    function add_upwind_drift_entries!(row, i, b_i)
        bplus = max(b_i, 0.0)
        bminus = min(b_i, 0.0)
        aL = bplus / Δk
        aU = (-bminus) / Δk

        push_entry!(row, row, aL + aU)
        if i > 1
            push_entry!(row, row - 1, -aL)
        end
        if i < Nk
            push_entry!(row, row + 1, -aU)
        end
        return nothing
    end

    invdt = inv(dt)
    exitI = p.σ1 + p.σ3 + p.μ
    exitC = p.αEpi + p.μ + p.σ2
    exitR = p.λ + p.μ

    @inbounds for i in 1:Nk
        rowS = idx(1, i)
        outS = pol.infection_rate[i] + pol.q_rate[i]
        push_entry!(rowS, rowS, p.ρ + invdt + outS)
        add_upwind_drift_entries!(rowS, i, -pol.bS[i])
        push_entry!(rowS, idx(2, i), -pol.infection_rate[i])
        push_entry!(rowS, idx(4, i), -pol.q_rate[i])
        rhs[rowS] = pol.uS[i] + invdt * Vnext.VS[i]

        rowI = idx(2, i)
        push_entry!(rowI, rowI, p.ρ + invdt + exitI)
        add_upwind_drift_entries!(rowI, i, -pol.bI[i])
        push_entry!(rowI, idx(1, i), -p.μ)
        push_entry!(rowI, idx(3, i), -p.σ1)
        push_entry!(rowI, idx(4, i), -p.σ3)
        rhs[rowI] = pol.uI[i] + invdt * Vnext.VI[i]

        rowC = idx(3, i)
        push_entry!(rowC, rowC, p.ρ + invdt + exitC)
        add_upwind_drift_entries!(rowC, i, -pol.bC[i])
        push_entry!(rowC, idx(1, i), -(p.αEpi + p.μ))
        push_entry!(rowC, idx(4, i), -p.σ2)
        rhs[rowC] = pol.uC[i] + invdt * Vnext.VC[i]

        rowR = idx(4, i)
        push_entry!(rowR, rowR, p.ρ + invdt + exitR)
        add_upwind_drift_entries!(rowR, i, -pol.bR[i])
        push_entry!(rowR, idx(1, i), -exitR)
        rhs[rowR] = pol.uR[i] + invdt * Vnext.VR[i]
    end

    return (M = sparse(I, J, X, nstate, nstate), rhs = rhs, policies = pol)
end

function dynamic_hjb_step(
    Vnext,
    Vn_guess,
    Fn,
    controls_old_n,
    w,
    r,
    LI,
    t,
    dt,
    p;
    deriv_cache=nothing,
)
    Viter = copy_value(Vn_guess)
    last_residual = Inf
    last_pol = nothing
    it_done = 0

    for it in 1:p.maxIterHJBDynamic
        sys = build_HJB_dynamic_step_system(
            Viter,
            Vnext,
            Fn,
            controls_old_n,
            w,
            r,
            LI,
            t,
            dt,
            p;
            deriv_cache=deriv_cache,
        )

        Vcandidate = unstack_value(sys.M \ sys.rhs, p)
        if !(all(isfinite, Vcandidate.VS) && all(isfinite, Vcandidate.VI) && all(isfinite, Vcandidate.VC) && all(isfinite, Vcandidate.VR))
            error("Non-finite values in dynamic HJB step at t=$t")
        end

        last_residual = max_value_distance(Vcandidate, Viter)
        ω = p.ωHJBDynamic
        Vnext_iter = (
            VS = (1 - ω) .* Viter.VS .+ ω .* Vcandidate.VS,
            VI = (1 - ω) .* Viter.VI .+ ω .* Vcandidate.VI,
            VC = (1 - ω) .* Viter.VC .+ ω .* Vcandidate.VC,
            VR = (1 - ω) .* Viter.VR .+ ω .* Vcandidate.VR,
        )

        Viter = Vnext_iter
        last_pol = sys.policies
        it_done = it

        if last_residual < p.tolHJBDynamic
            break
        end
    end

    if isnothing(last_pol)
        last_pol = compute_HJB_dynamic_policies(Viter, Fn, w, r, LI, t, p; deriv_cache=deriv_cache)
    end

    diagnostics = (
        iterations = it_done,
        residual = last_residual,
        min_q = minimum(last_pol.q_rate),
        max_q = maximum(last_pol.q_rate),
        min_bS = minimum(last_pol.bS),
        max_bS = maximum(last_pol.bS),
        min_bI = minimum(last_pol.bI),
        max_bI = maximum(last_pol.bI),
        min_bC = minimum(last_pol.bC),
        max_bC = maximum(last_pol.bC),
        min_bR = minimum(last_pol.bR),
        max_bR = maximum(last_pol.bR),
    )

    return Viter, diagnostics
end

function solve_hjb_backward(
    VT,
    F_path,
    controls_old_path,
    w_path,
    r_path,
    LI_path,
    p;
    V_guess_path=nothing,
    dt=nothing,
    t=nothing,
    show_progress=false,
)
    Nt = length(F_path)
    if Nt < 2
        error("F_path must contain at least two time nodes")
    end
    if length(w_path) != Nt || length(r_path) != Nt || length(LI_path) != Nt
        error("Price paths must have the same length as F_path")
    end

    dt_eff = isnothing(dt) ? p.Δt : dt
    if !(isfinite(dt_eff) && dt_eff > 0)
        error("dt must be finite and > 0")
    end

    t_path = isnothing(t) ? collect(range(0, p.T_End, length=Nt)) : t
    V_path = Vector{Any}(undef, Nt)
    V_path[end] = copy_value(VT)

    local_iterations = zeros(Int, Nt - 1)
    local_residuals = fill(NaN, Nt - 1)
    min_q = fill(NaN, Nt - 1)
    max_q = fill(NaN, Nt - 1)

    dcache = DerivDkCache(eltype(VT.VS), p.Nk)

    for n in (Nt - 1):-1:1
        Vguess = if isnothing(V_guess_path)
            copy_value(V_path[n + 1])
        else
            copy_value(V_guess_path[n])
        end

        Vn, diag = dynamic_hjb_step(
            V_path[n + 1],
            Vguess,
            F_path[n],
            controls_old_path[n],
            w_path[n],
            r_path[n],
            LI_path[n],
            t_path[n],
            dt_eff,
            p;
            deriv_cache=dcache,
        )

        V_path[n] = Vn
        local_iterations[n] = diag.iterations
        local_residuals[n] = diag.residual
        min_q[n] = diag.min_q
        max_q[n] = diag.max_q

        if show_progress
            print("\r\33[2Kdynamic HJB backward step $(Nt - n)/$(Nt - 1) | t=$(round(t_path[n], sigdigits=4)) | residual=$(round(diag.residual, sigdigits=3))")
            flush(stdout)
        end
    end

    show_progress && print("\n")
    diagnostics = (
        local_iterations = local_iterations,
        local_residuals = local_residuals,
        min_q = min_q,
        max_q = max_q,
    )

    return V_path, diagnostics
end
