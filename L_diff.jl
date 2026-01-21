


# Compute derivative for log(V'(k)) - central difference with positivity enforcement
function ∂k_log(f, p)
    return ∂k_log(f, p.Nk, p.Δk, p.ϵDkUp)
end

function ∂k_log(f, Nk, Δk, ϵ)
    """
    Compute safe derivative for log(V'(k)) using central differences
    """
    
    ∂f_log = similar(f)

    # k = 0
    Dp = (f[2] - f[1]) / Δk
    ∂f_log[1] = max(Dp, ϵ)

    # Interior points
    for i in 2:Nk-1
        Dm = (f[i]   - f[i-1]) / Δk
        Dp = (f[i+1] - f[i])   / Δk
        
        # Central difference with positivity enforcement
        ∂f_log[i] = max(0.5 * (Dm + Dp), ϵ)
    end

    # k = kmax
    Dm = (f[Nk] - f[Nk-1]) / Δk
    ∂f_log[Nk] = max(Dm, ϵ)

    return ∂f_log
end

# Compute upwind derivative for flux V'(k) * b(k)
function ∂k_flux(f, b, p)
    return ∂k_flux(f, b, p.Nk, p.Δk, p.ϵDkUp)
end

function ∂k_flux(f, b, Nk, Δk, ϵ)
    """
    Compute upwind derivative for V'(k) * b(k) flux term
    """
    
    ∂f_flux = similar(f)

    # k = 0
    Dp = (f[2] - f[1]) / Δk
    ∂f_flux[1] = max(Dp, ϵ)

    # Interior points - upwind scheme
    for i in 2:Nk-1
        Dm = (f[i]   - f[i-1]) / Δk
        Dp = (f[i+1] - f[i])   / Δk
        
        # Enforce positivity
        Dm = max(Dm, ϵ)
        Dp = max(Dp, ϵ)
        
        # Upwind: use backward if b >= 0, forward if b < 0
        ∂f_flux[i] = b[i] >= 0 ? Dm : Dp
    end

    # k = kmax
    Dm = (f[Nk] - f[Nk-1]) / Δk
    ∂f_flux[Nk] = max(Dm, ϵ)

    return ∂f_flux
end

# Legacy wrapper for backward compatibility
function ∂kHJB(f, b, p)
    return ∂kHJB(f, b, p.Nk, p.Δk, p.ϵDkUp)
end

function ∂kHJB(f, b, Nk, Δk, ϵ)
    """
    Compute discrete derivatives for HJB in 1D:
    - ∂f_log  : safe derivative for log(V'(k))
    - ∂f_flux : upwind derivative for V'(k) * b(k)
    """

    ∂f_log  = ∂k_log(f, Nk, Δk, ϵ)
    ∂f_flux = ∂k_flux(f, b, Nk, Δk, ϵ)

    return ∂f_log, ∂f_flux
end