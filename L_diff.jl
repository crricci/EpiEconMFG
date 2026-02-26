


# Safe derivative V'(k) (central difference + positivity floor)
"""
    ∂k_safe(f, p)

Compute a numerically safe approximation of ∂f/∂k on the capital grid.

Uses central differences in the interior and one-sided differences at the boundaries,
then floors the result at `p.ϵDkUp` to avoid division by (near) zero later on.
"""
function ∂k_safe(f, p)
    return ∂k_safe(f, p.Nk, p.Δk, p.ϵDkUp)
end

"""
    ∂k_safe!(∂f, f, Nk, Δk, ϵ)

In-place version of `∂k_safe`.

Computes a finite-difference derivative on a 1D grid with spacing `Δk`, using central
differences in the interior and one-sided differences at the boundaries, and floors
the derivative at `ϵ`.

`∂f` must be a vector of length `Nk`.
"""
function ∂k_safe!(∂f, f, Nk, Δk, ϵ)
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

"""
    ∂k_safe(f, Nk, Δk, ϵ)

Out-of-place version of `∂k_safe!`.

Allocates an output vector `∂f = similar(f)` and fills it with the safe finite-difference
derivative.
"""
function ∂k_safe(f, Nk, Δk, ϵ)
    ∂f = similar(f)
    return ∂k_safe!(∂f, f, Nk, Δk, ϵ)
end
