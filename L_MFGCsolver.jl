"""
    solveModel(p, F0; V0=nothing, verbose=p.verbose, HJB_every=1, save_stride=1,
              show_progress=nothing, progress_every=1)

Solve the coupled system: Forward Kolmogorov / Fokker–Planck (distribution dynamics)
and a stationary-form HJB (re-solved over time) using an implicit (Backward Euler)
time step for the FP equation.

The time grid is determined **only** by the parameters in `p`:
- `p.T_End`
- `p.Δt` (hence `Nstep = ceil(p.T_End/p.Δt)`)

# Arguments
- `p`: model parameters (`MFGEpiEcon`)
- `F0`: initial distribution either as a `NamedTuple` `(ϕSt, ϕIt, ϕCt, ϕRt)` (each
  a vector of length `p.Nk`) or as a stacked vector `phi0` of length `4*p.Nk`.

# Keyword Arguments
- `V0`: initial guess for the HJB value functions as `(VS, VI, VC, VR)` (default: zeros).
- `verbose`: overrides `p.verbose` *only during* this call.
- `HJB_every`: solve the HJB every `HJB_every` time steps (default: 1 = every step).
  If `HJB_every > 1`, the previous `V` and `w` are reused between HJB solves.
  The FP distribution is still updated every time step.
- `save_stride`: save outputs every `save_stride` time points (default: 1 = save all).
- `show_progress`: if `true`, prints progress even when `verbose=false`.
  Default: `!verbose`.
- `progress_every`: in `verbose=true` mode, updates the progress message every
  `progress_every` steps (default: 1).

# Returns
A `NamedTuple` with:
- `t`: saved time grid
- `F`: vector of distributions `F[n] = (ϕSt, ϕIt, ϕCt, ϕRt)`
- `V`: vector of HJB solutions `V[n] = (VS, VI, VC, VR, w, ...)`
- `controls`: vector of policy/auxiliary objects (consumption, labor, vaccination,
  drift, transition rates, prices/aggregates), each defined on the capital grid.

# Notes
- The HJB is time-independent in form, but is re-solved over time because it depends
  on the current distribution `F(t)` (via the infection externality).
- In non-verbose mode with progress enabled, a single line is continuously rewritten
  during the inner wage/HJB iterations, and a newline is printed once the time step ends.
  The update frequencies are controlled inside the parameter struct (e.g.
  `p.progressWage_every` and `p.progressHJB_every`).
"""
function solveModel(
    p,
    F0;
    V0 = nothing,
    verbose = p.verbose,
    HJB_every::Int = 1,
    save_stride::Int = 1,
    show_progress = nothing,
    progress_every::Int = 1,
)

    Nk = p.Nk
    if HJB_every < 1
        error("HJB_every must be >= 1")
    end
    if save_stride < 1
        error("save_stride must be >= 1")
    end
    if isnothing(show_progress)
        show_progress = !verbose
    end

    # Standardize initial distribution to stacked vector phi
    phi = if F0 isa AbstractVector
        if length(F0) != 4 * Nk
            error("If F0 is a vector, it must have length 4*Nk = $(4*Nk), got $(length(F0))")
        end
        copy(F0)
    else
        copy(stack_distribution(F0))
    end

    renormalize_distribution!(phi, p)

    # Time discretization
    T_End = p.T_End
    Δt = p.Δt
    if !(isfinite(T_End) && T_End > 0)
        error("p.T_End must be finite and > 0")
    end
    if !(isfinite(Δt) && Δt > 0)
        error("p.Δt must be finite and > 0")
    end
    Nstep_eff = Int(ceil(T_End / Δt))
    if Nstep_eff < 1
        error("Nstep must be >= 1")
    end

    t_full = collect(range(0, T_End, length=Nstep_eff + 1))
    Δt_eff = T_End / Nstep_eff

    # Initial guess for HJB
    Vguess = if isnothing(V0)
        (VS = zeros(eltype(phi), Nk), VI = zeros(eltype(phi), Nk), VC = zeros(eltype(phi), Nk), VR = zeros(eltype(phi), Nk))
    else
        (VS = copy(V0.VS), VI = copy(V0.VI), VC = copy(V0.VC), VR = copy(V0.VR))
    end

    dcache = DerivDkCache(eltype(Vguess.VS), Nk)

    # Decide which time indices to save (1-based indices over t_full)
    save_mask = falses(Nstep_eff + 1)
    for i in 1:save_stride:(Nstep_eff + 1)
        save_mask[i] = true
    end
    nsave = count(save_mask)

    # Pre-allocate saved output containers
    t = Vector{eltype(t_full)}(undef, nsave)
    Fts = Vector{Any}(undef, nsave)
    Vs = Vector{Any}(undef, nsave)
    controls = Vector{Any}(undef, nsave)
    save_ptr = 1

    # Temporarily override p.verbose if requested
    old_verbose = getproperty(p, :verbose)
    setproperty!(p, :verbose, verbose)

    lastVsol = nothing
    lastPol = nothing
    try
        for i in 1:Nstep_eff
            Ft = unstack_distribution(phi, p)

            do_solve = (i == 1) || ((i - 1) % HJB_every == 0)
            if do_solve || isnothing(lastVsol)
                if show_progress && !verbose
                    # Real-time progress: keep rewriting the SAME line (no newlines)
                    # while wage/HJB iterations run; then print exactly one newline when the
                    # time step finishes to "fix" that step's final line.
                    last_itw = 0
                    last_itV = 0
                    emit = function (itw, itV)
                        if itw > 0
                            last_itw = itw
                        end
                        if itV > 0
                            last_itV = itV
                        end
                        msg = "time step $(i)/$(Nstep_eff) | wage $(last_itw)/$(p.maxitWage) | HJB $(last_itV)/$(p.maxitHJBvalue)"
                        # Clear the line then rewrite it.
                        print("\r\33[2K", msg)
                        flush(stdout)
                        return nothing
                    end
                    emit(0, 0)
                    Vsol = value_iterationHJB(Vguess, Ft, p; progress_cb = emit)
                    emit(hasproperty(Vsol, :itw) ? Vsol.itw : last_itw, hasproperty(Vsol, :itV) ? Vsol.itV : last_itV)
                    print("\n")
                else
                    Vsol = value_iterationHJB(Vguess, Ft, p)
                end
                Vguess = (VS = Vsol.VS, VI = Vsol.VI, VC = Vsol.VC, VR = Vsol.VR)
                lastVsol = Vsol
            else
                Vsol = lastVsol
            end

            pol = compute_FP_policies((VS=Vsol.VS, VI=Vsol.VI, VC=Vsol.VC, VR=Vsol.VR), Ft, p; w=Vsol.w, deriv_cache=dcache)
            lastPol = pol

            # If HJB was reused, still print exactly one line for this time step.
            if show_progress && !verbose && !do_solve
                itw = hasproperty(Vsol, :itw) ? Vsol.itw : 0
                itV = hasproperty(Vsol, :itV) ? Vsol.itV : 0
                msg = "time step $(i)/$(Nstep_eff) | wage $(itw)/$(p.maxitWage) | HJB $(itV)/$(p.maxitHJBvalue)"
                print("\r\33[2K", msg, "\n")
            end

            # If we are not in real-time mode, print one progress line for this time step.
            if show_progress && verbose && (i == 1 || i == Nstep_eff || (progress_every > 0 && i % progress_every == 0))
                itw = hasproperty(Vsol, :itw) ? Vsol.itw : 0
                itV = hasproperty(Vsol, :itV) ? Vsol.itV : 0
                msg = "time step $(i)/$(Nstep_eff) | wage $(itw)/$(p.maxitWage) | HJB $(itV)/$(p.maxitHJBvalue)"
                print("\r", rpad(msg, 110))
                flush(stdout)
            end

            if save_mask[i]
                t[save_ptr] = t_full[i]
                Fts[save_ptr] = Ft
                Vs[save_ptr] = Vsol
                controls[save_ptr] = pol
                save_ptr += 1
            end

            # Build generator and do one implicit Euler step for the distribution
            G = build_FP_generator(pol, p)
            nstate = 4 * Nk
            A = spdiagm(0 => ones(eltype(phi), nstate)) - Δt_eff * G
            phi = lu(A) \ phi

            project_nonnegative!(phi)
            renormalize_distribution!(phi, p)
        end

        # Final time: optionally store at t_end
        FtT = unstack_distribution(phi, p)
        if save_mask[end]
            VsolT = value_iterationHJB(Vguess, FtT, p)
            polT = compute_FP_policies((VS=VsolT.VS, VI=VsolT.VI, VC=VsolT.VC, VR=VsolT.VR), FtT, p; w=VsolT.w, deriv_cache=dcache)
            t[save_ptr] = t_full[end]
            Fts[save_ptr] = FtT
            Vs[save_ptr] = VsolT
            controls[save_ptr] = polT
            # save_ptr += 1
        end
        show_progress && verbose && print("\n")
    finally
        setproperty!(p, :verbose, old_verbose)
    end

    return (t = t, F = Fts, V = Vs, controls = controls)
end
