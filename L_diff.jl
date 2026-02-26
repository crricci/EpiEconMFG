


# Safe derivative V'(k) (central difference + positivity floor)
function ∂k_safe(f, p)
    return ∂k_safe(f, p.Nk, p.Δk, p.ϵDkUp)
end

function ∂k_safe!(∂f, f, Nk, Δk, ϵ)
    """
    In-place safe derivative for V'(k) using central differences.
    Floors the derivative at ϵ to avoid division by (near) zero.

    `∂f` must be a vector of length Nk.
    """

    @inbounds begin
        # k = 0
        Dp = (f[2] - f[1]) / Δk
        ∂f[1] = max(Dp, ϵ)

        # Interior points
        for i in 2:Nk-1
            Dm = (f[i]   - f[i-1]) / Δk
            Dp = (f[i+1] - f[i])   / Δk
            ∂f[i] = max(0.5 * (Dm + Dp), ϵ)
        end

        # k = kmax
        Dm = (f[Nk] - f[Nk-1]) / Δk
        ∂f[Nk] = max(Dm, ϵ)
    end

    return ∂f
end

function ∂k_safe(f, Nk, Δk, ϵ)
    """
    Compute safe derivative V'(k) using central differences.
    """

    ∂f = similar(f)
    return ∂k_safe!(∂f, f, Nk, Δk, ϵ)
end
