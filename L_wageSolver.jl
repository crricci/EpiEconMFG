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
    # Define residual function: g(w) = 0 at fixed point
    g(w) = T_wage(w, V, ∂V, F, p) - w
    
    # Solve using root finding method
    w_star = w0  # Initialize with default value
    try
        w_star = find_zero(g, w0, verbose = false)
    catch e
        # If convergence failed, keep the input value
        if p.verbose
            println("Warning: Fixed point solver failed to converge. Keeping w = $w0")
            println("Error: $e")
        end
    end
    
    return w_star
end


function T_wage(w, V, ∂V, F, p)
    """ given wage computes L and K, then updates wage using the production function """


    lOpt, WEff = optimal_labor_ALL(V, ∂V, F, w, p)

    L = aggregate_labor_supply(lOpt, F, p)
    K = aggregate_kapital(F, p)

    # Update wage using the production function
    w_new = wage(K, L, p)

    return w_new
end

