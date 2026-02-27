using CairoMakie

function _time_by_k_matrix(get_vec, Nt::Int, Nk::Int)
	v1 = get_vec(1)
	T = eltype(v1)
	Z = Matrix{T}(undef, Nt, Nk)
	@inbounds Z[1, :] .= v1
	@inbounds for n in 2:Nt
		Z[n, :] .= get_vec(n)
	end
	return Z
end

function _global_minmax(mats...)
	lo = Inf
	hi = -Inf
	for M in mats
		mlo = minimum(M)
		mhi = maximum(M)
		lo = min(lo, mlo)
		hi = max(hi, mhi)
	end
	return lo, hi
end

function _safe_colorrange(lo, hi)
	if !(isfinite(lo) && isfinite(hi))
		return (0.0, 1.0)
	end
	if hi > lo
		return (Float64(lo), Float64(hi))
	end
	δ = eps(Float64(max(abs(lo), 1.0)))
	return (Float64(lo), Float64(lo + δ))
end

function _heatmap_with_contours!(ax, t, k, Z_tk;
	colormap = :viridis,
	colorrange,
	contour_lines::Int = 6,
	contour_color = :black,
	contour_linewidth = 0.6,
)
	hm = CairoMakie.heatmap!(ax, t, k, Z_tk; colormap = colormap, colorrange = colorrange)
	lo, hi = colorrange
	if contour_lines > 0 && isfinite(lo) && isfinite(hi) && (hi > lo)
		levels = range(lo, hi; length = contour_lines + 2)[2:end-1]
		CairoMakie.contour!(ax, t, k, Z_tk; levels = levels, color = contour_color, linewidth = contour_linewidth)
	end
	return hm
end

function save_figure_1_totals(result, p;
	outdir::AbstractString = "figures",
	filename::AbstractString = "figure_1_totals_SICR_over_time.pdf",
)
	mkpath(outdir)

	t = result.t
	Nt = length(t)
	if Nt == 0
		error("result.t is empty")
	end
	if length(result.F) != Nt
		error("Inconsistent result lengths: length(t)=$Nt, length(F)=$(length(result.F))")
	end

	S_tot = zeros(Float64, Nt)
	I_tot = zeros(Float64, Nt)
	C_tot = zeros(Float64, Nt)
	R_tot = zeros(Float64, Nt)
	@inbounds for n in 1:Nt
		Ft = result.F[n]
		S_tot[n] = sum(Ft.ϕSt) * p.Δk
		I_tot[n] = sum(Ft.ϕIt) * p.Δk
		C_tot[n] = sum(Ft.ϕCt) * p.Δk
		R_tot[n] = sum(Ft.ϕRt) * p.Δk
	end

	fig = CairoMakie.Figure(size = (900, 450))
	ax = CairoMakie.Axis(fig[1, 1], xlabel = "t", ylabel = "mass")
	CairoMakie.lines!(ax, t, S_tot, label = "S")
	CairoMakie.lines!(ax, t, I_tot, label = "I")
	CairoMakie.lines!(ax, t, C_tot, label = "C")
	CairoMakie.lines!(ax, t, R_tot, label = "R")
	CairoMakie.axislegend(ax, position = :rt)
	CairoMakie.save(joinpath(outdir, filename), fig)
	return nothing
end

function save_figure_2_distributions(result, p;
	outdir::AbstractString = "figures",
	filename::AbstractString = "figure_2_heatmaps_distributions_SICR_tk.pdf",
	contour_lines::Int = 6,
)
	mkpath(outdir)

	t = result.t
	Nt = length(t)
	Nk = p.Nk
	k = collect(p.k)
	if Nt == 0
		error("result.t is empty")
	end
	if length(result.F) != Nt
		error("Inconsistent result lengths: length(t)=$Nt, length(F)=$(length(result.F))")
	end

	ΦS = _time_by_k_matrix(n -> result.F[n].ϕSt, Nt, Nk)
	ΦI = _time_by_k_matrix(n -> result.F[n].ϕIt, Nt, Nk)
	ΦC = _time_by_k_matrix(n -> result.F[n].ϕCt, Nt, Nk)
	ΦR = _time_by_k_matrix(n -> result.F[n].ϕRt, Nt, Nk)

	ϕ_hi = maximum((maximum(ΦS), maximum(ΦI), maximum(ΦC), maximum(ΦR)))
	clims_ϕ = _safe_colorrange(0.0, ϕ_hi)

	fig = CairoMakie.Figure(size = (1200, 800))
	grid = CairoMakie.GridLayout()
	fig[1, 1] = grid

	axS = CairoMakie.Axis(grid[1, 1], title = "S(t,k)", xlabel = "t", ylabel = "k")
	axI = CairoMakie.Axis(grid[1, 2], title = "I(t,k)", xlabel = "t", ylabel = "k")
	axC = CairoMakie.Axis(grid[2, 1], title = "C(t,k)", xlabel = "t", ylabel = "k")
	axR = CairoMakie.Axis(grid[2, 2], title = "R(t,k)", xlabel = "t", ylabel = "k")

	hmS = _heatmap_with_contours!(axS, t, k, ΦS; colormap = :viridis, colorrange = clims_ϕ, contour_lines = contour_lines)
	_heatmap_with_contours!(axI, t, k, ΦI; colormap = :viridis, colorrange = clims_ϕ, contour_lines = contour_lines)
	_heatmap_with_contours!(axC, t, k, ΦC; colormap = :viridis, colorrange = clims_ϕ, contour_lines = contour_lines)
	_heatmap_with_contours!(axR, t, k, ΦR; colormap = :viridis, colorrange = clims_ϕ, contour_lines = contour_lines)

	CairoMakie.Colorbar(fig[1, 2], hmS)
	CairoMakie.save(joinpath(outdir, filename), fig)
	return nothing
end

function save_figure_3_flux_S_to_I(result, p;
	outdir::AbstractString = "figures",
	filename::AbstractString = "figure_3_heatmap_flux_S_to_I_tk.pdf",
	contour_lines::Int = 6,
)
	mkpath(outdir)

	t = result.t
	Nt = length(t)
	Nk = p.Nk
	k = collect(p.k)
	if Nt == 0
		error("result.t is empty")
	end
	if length(result.F) != Nt || length(result.controls) != Nt
		error("Inconsistent result lengths: length(t)=$Nt, length(F)=$(length(result.F)), length(controls)=$(length(result.controls))")
	end

	FluxSI = _time_by_k_matrix(n -> (result.controls[n].infection_rate .* result.F[n].ϕSt), Nt, Nk)
	flux_hi = maximum(FluxSI)
	clims_flux = _safe_colorrange(0.0, flux_hi)

	fig = CairoMakie.Figure(size = (1100, 650))
	ax = CairoMakie.Axis(fig[1, 1], title = "Flow S→I: β lS*(t,k) ϕS(t,k) LI(t)", xlabel = "t", ylabel = "k")
	hm = _heatmap_with_contours!(ax, t, k, FluxSI; colormap = :viridis, colorrange = clims_flux, contour_lines = contour_lines)
	CairoMakie.Colorbar(fig[1, 2], hm)
	CairoMakie.save(joinpath(outdir, filename), fig)
	return nothing
end

function save_figure_4_consumption(result, p;
	outdir::AbstractString = "figures",
	filename::AbstractString = "figure_4_heatmaps_consumption_SICR_tk.pdf",
	contour_lines::Int = 6,
)
	mkpath(outdir)

	t = result.t
	Nt = length(t)
	Nk = p.Nk
	k = collect(p.k)
	if Nt == 0
		error("result.t is empty")
	end
	if length(result.controls) != Nt
		error("Inconsistent result lengths: length(t)=$Nt, length(controls)=$(length(result.controls))")
	end

	CS = _time_by_k_matrix(n -> result.controls[n].cS, Nt, Nk)
	CI = _time_by_k_matrix(n -> result.controls[n].cI, Nt, Nk)
	CC = _time_by_k_matrix(n -> result.controls[n].cC, Nt, Nk)
	CR = _time_by_k_matrix(n -> result.controls[n].cR, Nt, Nk)

	c_lo, c_hi = _global_minmax(CS, CI, CC, CR)
	clims_c = _safe_colorrange(c_lo, c_hi)

	fig = CairoMakie.Figure(size = (1200, 800))
	grid = CairoMakie.GridLayout()
	fig[1, 1] = grid

	axS = CairoMakie.Axis(grid[1, 1], title = "cS(t,k)", xlabel = "t", ylabel = "k")
	axI = CairoMakie.Axis(grid[1, 2], title = "cI(t,k)", xlabel = "t", ylabel = "k")
	axC = CairoMakie.Axis(grid[2, 1], title = "cC(t,k)", xlabel = "t", ylabel = "k")
	axR = CairoMakie.Axis(grid[2, 2], title = "cR(t,k)", xlabel = "t", ylabel = "k")

	hmS = _heatmap_with_contours!(axS, t, k, CS; colormap = :plasma, colorrange = clims_c, contour_lines = contour_lines)
	_heatmap_with_contours!(axI, t, k, CI; colormap = :plasma, colorrange = clims_c, contour_lines = contour_lines)
	_heatmap_with_contours!(axC, t, k, CC; colormap = :plasma, colorrange = clims_c, contour_lines = contour_lines)
	_heatmap_with_contours!(axR, t, k, CR; colormap = :plasma, colorrange = clims_c, contour_lines = contour_lines)

	CairoMakie.Colorbar(fig[1, 2], hmS)
	CairoMakie.save(joinpath(outdir, filename), fig)
	return nothing
end

function save_figure_5_labor(result, p;
	outdir::AbstractString = "figures",
	filename::AbstractString = "figure_5_heatmaps_labor_SIR_tk.pdf",
	contour_lines::Int = 6,
)
	mkpath(outdir)

	t = result.t
	Nt = length(t)
	Nk = p.Nk
	k = collect(p.k)
	if Nt == 0
		error("result.t is empty")
	end
	if length(result.controls) != Nt
		error("Inconsistent result lengths: length(t)=$Nt, length(controls)=$(length(result.controls))")
	end

	LS = _time_by_k_matrix(n -> result.controls[n].lOpt.lS, Nt, Nk)
	LI = _time_by_k_matrix(n -> result.controls[n].lOpt.lI, Nt, Nk)
	LR = _time_by_k_matrix(n -> result.controls[n].lOpt.lR, Nt, Nk)
	clims_l = (0.0, 1.0)

	fig = CairoMakie.Figure(size = (1400, 500))
	grid = CairoMakie.GridLayout()
	fig[1, 1] = grid

	axS = CairoMakie.Axis(grid[1, 1], title = "lS(t,k)", xlabel = "t", ylabel = "k")
	axI = CairoMakie.Axis(grid[1, 2], title = "lI(t,k)", xlabel = "t", ylabel = "k")
	axR = CairoMakie.Axis(grid[1, 3], title = "lR(t,k)", xlabel = "t", ylabel = "k")

	hmS = _heatmap_with_contours!(axS, t, k, LS; colormap = :viridis, colorrange = clims_l, contour_lines = contour_lines)
	_heatmap_with_contours!(axI, t, k, LI; colormap = :viridis, colorrange = clims_l, contour_lines = contour_lines)
	_heatmap_with_contours!(axR, t, k, LR; colormap = :viridis, colorrange = clims_l, contour_lines = contour_lines)

	CairoMakie.Colorbar(fig[1, 2], hmS)
	CairoMakie.save(joinpath(outdir, filename), fig)
	return nothing
end

function save_figure_6_vaccination_q(result, p;
	outdir::AbstractString = "figures",
	filename::AbstractString = "figure_6_heatmap_q_tk.pdf",
	contour_lines::Int = 6,
)
	mkpath(outdir)

	t = result.t
	Nt = length(t)
	Nk = p.Nk
	k = collect(p.k)
	if Nt == 0
		error("result.t is empty")
	end
	if length(result.controls) != Nt
		error("Inconsistent result lengths: length(t)=$Nt, length(controls)=$(length(result.controls))")
	end

	Q = _time_by_k_matrix(n -> result.controls[n].q_rate, Nt, Nk)
	q_hi = maximum(Q)
	clims_q = _safe_colorrange(0.0, q_hi)

	fig = CairoMakie.Figure(size = (1100, 650))
	ax = CairoMakie.Axis(fig[1, 1], title = "Vaccination intensity q(t,k)", xlabel = "t", ylabel = "k")
	hm = _heatmap_with_contours!(ax, t, k, Q; colormap = :magma, colorrange = clims_q, contour_lines = contour_lines)
	CairoMakie.Colorbar(fig[1, 2], hm)
	CairoMakie.save(joinpath(outdir, filename), fig)
	return nothing
end

"""
	save_all_figures(result, p; outdir="figures", contour_lines=6)

Generate and save a set of PDF figures from the output of `solveModel`.

Each figure is produced by a separate `save_figure_*` function.

Output filenames use the `figure_#_...` prefix (e.g. `figure_1_...`).
Heatmaps are grouped into multi-panel figures (2×2 or 1×3) and include a few
contour lines overlaid to improve readability.
"""
function save_all_figures(result, p;
	outdir::AbstractString = "figures",
	contour_lines::Int = 6,
)
	mkpath(outdir)

	save_figure_1_totals(result, p; outdir = outdir)
	save_figure_2_distributions(result, p; outdir = outdir, contour_lines = contour_lines)
	save_figure_3_flux_S_to_I(result, p; outdir = outdir, contour_lines = contour_lines)
	save_figure_4_consumption(result, p; outdir = outdir, contour_lines = contour_lines)
	save_figure_5_labor(result, p; outdir = outdir, contour_lines = contour_lines)
	save_figure_6_vaccination_q(result, p; outdir = outdir, contour_lines = contour_lines)

	return nothing
end

