


# Compute derivative for log(V'(k)) - central difference with positivity enforcement
function ∂k_log(f, p)
    return ∂k_log(f, p.Nk, p.Δk, p.ϵDkUp)
end

function ∂k_log!(∂f_log, f, Nk, Δk, ϵ)
    """
    In-place safe derivative for V'(k) using central differences.
    Floors the derivative at ϵ to avoid division by (near) zero.

    `∂f_log` must be a vector of length Nk.
    """

    @inbounds begin
        # k = 0
        Dp = (f[2] - f[1]) / Δk
        ∂f_log[1] = max(Dp, ϵ)

        # Interior points
        for i in 2:Nk-1
            Dm = (f[i]   - f[i-1]) / Δk
            Dp = (f[i+1] - f[i])   / Δk
            ∂f_log[i] = max(0.5 * (Dm + Dp), ϵ)
        end

        # k = kmax
        Dm = (f[Nk] - f[Nk-1]) / Δk
        ∂f_log[Nk] = max(Dm, ϵ)
    end

    return ∂f_log
end

function ∂k_log(f, Nk, Δk, ϵ)
    """
    Compute safe derivative for log(V'(k)) using central differences
    """

    ∂f_log = similar(f)
    return ∂k_log!(∂f_log, f, Nk, Δk, ϵ)
end
