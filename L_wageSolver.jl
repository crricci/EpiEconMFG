function fixed_point_wage(V, ∂V, F, p; w0)
    """
    Solves: w = T_wage(w, V, F, p)

    Given wages find optimal labour allocation, aggregate labor and capital, 
    then update wage using the production function. Iterate until convergence.
    
    Arguments:
    - V, ∂V, F, p: parameters passed to T_wage
    - w0: initial guess for wage
    
    Returns: equilibrium wage w*
    """
    w_init = isfinite(w0) ? max(w0, p.ϵDkUp) : max(p.w_start, p.ϵDkUp)

    # Residual: g(w) = T_wage(w) - w
    g(w) = T_wage(w, V, ∂V, F, p) - w

    # 1) Try bracketed bisection (very robust)
    w_low = max(w_init / 10.0, p.ϵDkUp)
    w_high = w_init * 10.0
    g_low = NaN
    g_high = NaN
    bracket_found = false

    for _ in 1:25
        g_low = g(w_low)
        g_high = g(w_high)

        if isfinite(g_low) && isfinite(g_high) && (sign(g_low) != sign(g_high))
            bracket_found = true
            break
        end

        # Expand bracket geometrically
        w_low = max(w_low / 2.0, p.ϵDkUp)
        w_high *= 2.0
    end

    if bracket_found
        try
            w_star = find_zero(g, (w_low, w_high), Bisection(); atol = 1e-10, rtol = 1e-10)
            return isfinite(w_star) ? max(w_star, p.ϵDkUp) : w_init
        catch e
            p.verbose && println("Warning: Bisection wage solve failed ($e). Falling back to fixed-point iteration.")
        end
    end

    # 2) Fallback: damped fixed-point iteration w <- (1-ξ)w + ξ T_wage(w)
    ξ = 0.2
    w = w_init
    for it in 1:100
        w_new = T_wage(w, V, ∂V, F, p)
        if !(isfinite(w_new) && w_new > 0.0)
            p.verbose && println("Warning: T_wage returned non-finite or non-positive wage at it=$it. Keeping w=$w")
            return w
        end

        w_next = (1 - ξ) * w + ξ * w_new
        if abs(w_next - w) <= 1e-10 * max(1.0, abs(w))
            return max(w_next, p.ϵDkUp)
        end
        w = w_next
    end

    p.verbose && println("Warning: Fixed-point wage iteration did not converge. Keeping w=$w")
    return max(w, p.ϵDkUp)
end


function T_wage(w, V, ∂V, F, p)
    """ given wage computes L and K, then updates wage using the production function """


    lOpt, _ = optimal_labor_ALL(V, ∂V, F, w, p)

    L = aggregate_labor_supply(lOpt, F, p)
    K = aggregate_kapital(F, p)

    # Update wage using the production function
    w_new = wage(K, L, p)

    return w_new
end

