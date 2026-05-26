using CairoMakie
using ProgressMeter

function _save_as_jpg(path::AbstractString, fig;
	px_per_unit::Real = 1.35,
	quality::Int = 92,
)
	# CairoMakie does not support image/jpeg directly; on macOS we convert via `sips`.
	# We still try direct save first, in case future versions support it.
	try
		CairoMakie.save(path, fig; px_per_unit = px_per_unit)
		return path
	catch
		# fall back below
	end

	tmp_png = tempname() * ".png"
	CairoMakie.save(tmp_png, fig; px_per_unit = px_per_unit)

	sips = Sys.which("sips")
	if Sys.isapple() && sips !== nothing
		cmd = `$sips -s format jpeg -s formatOptions $quality $tmp_png --out $path`
		try
			run(pipeline(cmd, stdout = devnull, stderr = devnull))
			rm(tmp_png; force = true)
			return path
		catch
			# continue to PNG fallback
		end
	end

	png_path = replace(path, r"\.(jpe?g)$"i => ".png")
	mv(tmp_png, png_path; force = true)
	return png_path
end

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

function _plot_k_indices(result, p;
	truncate_k_to_initial_mass::Bool = true,
	k_mass_level::Real = 0.9,
)
	if !truncate_k_to_initial_mass
		return collect(1:p.Nk)
	end
	if !(0 < k_mass_level <= 1)
		error("k_mass_level must be in (0, 1], got $k_mass_level")
	end
	if !haskey(result, :F) || isempty(result.F)
		error("Cannot compute k plotting range: result.F is missing or empty")
	end

	F0 = result.F[1]
	mass_density0 = F0.ϕSt .+ F0.ϕIt .+ F0.ϕCt .+ F0.ϕRt
	total_mass0 = sum(mass_density0) * p.Δk
	if !(isfinite(total_mass0) && total_mass0 > 0)
		error("Cannot compute k plotting range: non-positive initial mass $total_mass0")
	end

	cumulative = 0.0
	last_idx = p.Nk
	@inbounds for i in 1:p.Nk
		cumulative += mass_density0[i] * p.Δk
		if cumulative / total_mass0 >= k_mass_level
			last_idx = i
			break
		end
	end

	return collect(1:max(1, last_idx))
end

function _restrict_k(k, k_indices, mats...)
	out = Any[k[k_indices]]
	for M in mats
		push!(out, M[:, k_indices])
	end
	return tuple(out...)
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

function _finite_minmax(M)
	lo = Inf
	hi = -Inf
	@inbounds for j in axes(M, 2)
		for i in axes(M, 1)
			v = Float64(M[i, j])
			if isfinite(v)
				lo = min(lo, v)
				hi = max(hi, v)
			end
		end
	end
	return lo, hi
end

function _format_sci(x::Real; sigdigits::Int = 2)
	v = Float64(x)
	if !isfinite(v)
		return "NaN"
	end
	if v == 0.0
		return "0"
	end
	ax = abs(v)
	exp10 = floor(Int, log10(ax))
	mant = v / (10.0 ^ exp10)
	# keep a short mantissa; use general format so 3.40 -> 3.4
	mant_str = string(round(mant; sigdigits = sigdigits))
	return string(mant_str, "*10^", exp10)
end

function _default_contour_labelformatter(level)
	v = Float64(level)
	if v == 0.0
		return "0"
	end
	av = abs(v)
	# Scientific notation for small/large values; otherwise plain.
	if av < 1e-2 || av >= 1e3
		return _format_sci(v)
	end
	return string(round(v; sigdigits = 3))
end

function _subsample_indices(N::Int, maxN::Int)
	if N <= maxN
		return collect(1:N)
	end
	# Round may introduce duplicates; unique keeps order.
	idx = unique(round.(Int, range(1, N; length = maxN)))
	# Ensure endpoints.
	if first(idx) != 1
		idx[1] = 1
	end
	if last(idx) != N
		idx[end] = N
	end
	return idx
end

function _surface_plot!(ax3, t, k, Z_tk;
	colormap = :viridis,
	colorrange,
	alpha::Float64 = 1.0,
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
)
	Nt, Nk = size(Z_tk)
	ti = _subsample_indices(Nt, maxNt)
	ki = _subsample_indices(Nk, maxNk)
	t2 = t[ti]
	k2 = k[ki]
	Z2 = Z_tk[ti, ki]

	# Surface: x=t, y=k, z=Z(t,k) with color mapped to Z
	plt = CairoMakie.surface!(ax3, t2, k2, Z2;
		colormap = colormap,
		colorrange = colorrange,
		color = Z2,
		transparency = alpha < 1.0,
		alpha = alpha,
		rasterize = rasterize,
	)
	return plt
end

function _heatmap_with_contours!(ax, t, k, Z_tk;
	colormap = :viridis,
	colorrange,
	contour_lines::Int = 6,
	contour_labels::Bool = true,
	contour_labelsize = 9,
	contour_labelformatter = _default_contour_labelformatter,
	contour_color = :black,
	contour_linewidth = 0.6,
)
	hm = CairoMakie.heatmap!(ax, t, k, Z_tk; colormap = colormap, colorrange = colorrange)
	cr_lo, cr_hi = colorrange
	data_lo, data_hi = _finite_minmax(Z_tk)
	lo = max(cr_lo, data_lo)
	hi = min(cr_hi, data_hi)
	if contour_lines > 0 && isfinite(lo) && isfinite(hi) && (hi > lo)
		levels = range(lo, hi; length = contour_lines + 2)[2:end-1]
		try
			CairoMakie.contour!(ax, t, k, Z_tk;
				levels = levels,
				color = contour_color,
				linewidth = contour_linewidth,
				labels = contour_labels,
				labelsize = contour_labelsize,
				labelcolor = contour_color,
				labelformatter = contour_labelformatter,
			)
		catch
			# Makie can error when labels are requested but no contour segments exist.
			CairoMakie.contour!(ax, t, k, Z_tk;
				levels = levels,
				color = contour_color,
				linewidth = contour_linewidth,
				labels = false,
			)
		end
	end
	return hm
end

function save_figure_1_totals(result, p;
	outdir::AbstractString = "outputs/figures",
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
	if !haskey(result, :controls) || length(result.controls) != Nt
		error("Figure 1 requires result.controls with length Nt=$Nt to compute vaccination flow")
	end

	S_tot = zeros(Float64, Nt)
	I_tot = zeros(Float64, Nt)
	C_tot = zeros(Float64, Nt)
	R_tot = zeros(Float64, Nt)
	Vax_flow = zeros(Float64, Nt)
	@inbounds for n in 1:Nt
		Ft = result.F[n]
		S_tot[n] = sum(Ft.ϕSt) * p.Δk
		I_tot[n] = sum(Ft.ϕIt) * p.Δk
		C_tot[n] = sum(Ft.ϕCt) * p.Δk
		R_tot[n] = sum(Ft.ϕRt) * p.Δk
		Vax_flow[n] = sum(result.controls[n].q_rate .* Ft.ϕSt) * p.Δk
	end

	fig = CairoMakie.Figure(size = (900, 450))
	ax = CairoMakie.Axis(fig[1, 1], xlabel = "t", ylabel = "Population share")
	CairoMakie.lines!(ax, t, S_tot, label = "Susceptible")
	CairoMakie.lines!(ax, t, I_tot, label = "Infected")
	CairoMakie.lines!(ax, t, C_tot, label = "Contained")
	CairoMakie.lines!(ax, t, R_tot, label = "Recovered")
	CairoMakie.lines!(ax, t, Vax_flow, label = "Vaccination flow ∫ q·S dk")
	CairoMakie.axislegend(ax, position = :rt)
	CairoMakie.save(joinpath(outdir, filename), fig)
	return nothing
end

function save_figure_2_distributions(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_2_heatmaps_distributions_SICR_tk.pdf",
	contour_lines::Int = 6,
	k_indices = nothing,
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
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, ΦS, ΦI, ΦC, ΦR = _restrict_k(k, kidx, ΦS, ΦI, ΦC, ΦR)

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

function save_figure_2Rel_relative_shares(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_2Rel_heatmaps_relative_shares_SICR_tk.pdf",
	contour_lines::Int = 6,
	k_indices = nothing,
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
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, ΦS, ΦI, ΦC, ΦR = _restrict_k(k, kidx, ΦS, ΦI, ΦC, ΦR)

	den = ΦS .+ ΦI .+ ΦC .+ ΦR
	RS = similar(den, Float64)
	RI = similar(den, Float64)
	RC = similar(den, Float64)
	RR = similar(den, Float64)
	@inbounds for j in axes(den, 2)
		for i in axes(den, 1)
			d = Float64(den[i, j])
			if isfinite(d) && d > 0
				RS[i, j] = Float64(ΦS[i, j]) / d
				RI[i, j] = Float64(ΦI[i, j]) / d
				RC[i, j] = Float64(ΦC[i, j]) / d
				RR[i, j] = Float64(ΦR[i, j]) / d
			else
				RS[i, j] = 0.0
				RI[i, j] = 0.0
				RC[i, j] = 0.0
				RR[i, j] = 0.0
			end
		end
	end

	clims = (0.0, 1.0)
	fig = CairoMakie.Figure(size = (1200, 800))
	grid = CairoMakie.GridLayout()
	fig[1, 1] = grid

	axS = CairoMakie.Axis(grid[1, 1], title = "S/(S+I+C+R)", xlabel = "t", ylabel = "k")
	axI = CairoMakie.Axis(grid[1, 2], title = "I/(S+I+C+R)", xlabel = "t", ylabel = "k")
	axC = CairoMakie.Axis(grid[2, 1], title = "C/(S+I+C+R)", xlabel = "t", ylabel = "k")
	axR = CairoMakie.Axis(grid[2, 2], title = "R/(S+I+C+R)", xlabel = "t", ylabel = "k")

	hmS = _heatmap_with_contours!(axS, t, k, RS; colormap = :viridis, colorrange = clims, contour_lines = contour_lines)
	_heatmap_with_contours!(axI, t, k, RI; colormap = :viridis, colorrange = clims, contour_lines = contour_lines)
	_heatmap_with_contours!(axC, t, k, RC; colormap = :viridis, colorrange = clims, contour_lines = contour_lines)
	_heatmap_with_contours!(axR, t, k, RR; colormap = :viridis, colorrange = clims, contour_lines = contour_lines)

	CairoMakie.Colorbar(fig[1, 2], hmS)
	CairoMakie.save(joinpath(outdir, filename), fig)
	return nothing
end

function save_figure_2bis_distributions_surface(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_2bis_surfaces_distributions_SICR_tk.png",
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
	px_per_unit::Real = 1.35,
	jpg_quality::Int = 92,
	k_indices = nothing,
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
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, ΦS, ΦI, ΦC, ΦR = _restrict_k(k, kidx, ΦS, ΦI, ΦC, ΦR)

	ϕ_hi = maximum((maximum(ΦS), maximum(ΦI), maximum(ΦC), maximum(ΦR)))
	clims_ϕ = _safe_colorrange(0.0, ϕ_hi)

	fig = CairoMakie.Figure(size = (1100, 750))
	grid = CairoMakie.GridLayout()
	fig[1, 1] = grid

	axS = CairoMakie.Axis3(grid[1, 1], title = "S(t,k)", xlabel = "t", ylabel = "k", zlabel = "density")
	axI = CairoMakie.Axis3(grid[1, 2], title = "I(t,k)", xlabel = "t", ylabel = "k", zlabel = "density")
	axC = CairoMakie.Axis3(grid[2, 1], title = "C(t,k)", xlabel = "t", ylabel = "k", zlabel = "density")
	axR = CairoMakie.Axis3(grid[2, 2], title = "R(t,k)", xlabel = "t", ylabel = "k", zlabel = "density")

	pltS = _surface_plot!(axS, t, k, ΦS; colormap = :viridis, colorrange = clims_ϕ, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)
	_surface_plot!(axI, t, k, ΦI; colormap = :viridis, colorrange = clims_ϕ, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)
	_surface_plot!(axC, t, k, ΦC; colormap = :viridis, colorrange = clims_ϕ, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)
	_surface_plot!(axR, t, k, ΦR; colormap = :viridis, colorrange = clims_ϕ, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)

	CairoMakie.Colorbar(fig[1, 2], pltS)
	outpath = joinpath(outdir, filename)
	if endswith(lowercase(filename), ".jpg") || endswith(lowercase(filename), ".jpeg")
		_save_as_jpg(outpath, fig; px_per_unit = px_per_unit, quality = jpg_quality)
	else
		CairoMakie.save(outpath, fig)
	end
	return nothing
end

function save_figure_2Relbis_relative_shares_surface(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_2Relbis_surfaces_relative_shares_SICR_tk.png",
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
	px_per_unit::Real = 1.35,
	jpg_quality::Int = 92,
	k_indices = nothing,
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
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, ΦS, ΦI, ΦC, ΦR = _restrict_k(k, kidx, ΦS, ΦI, ΦC, ΦR)

	den = ΦS .+ ΦI .+ ΦC .+ ΦR
	RS = similar(den, Float64)
	RI = similar(den, Float64)
	RC = similar(den, Float64)
	RR = similar(den, Float64)
	@inbounds for j in axes(den, 2)
		for i in axes(den, 1)
			d = Float64(den[i, j])
			if isfinite(d) && d > 0
				RS[i, j] = Float64(ΦS[i, j]) / d
				RI[i, j] = Float64(ΦI[i, j]) / d
				RC[i, j] = Float64(ΦC[i, j]) / d
				RR[i, j] = Float64(ΦR[i, j]) / d
			else
				RS[i, j] = 0.0
				RI[i, j] = 0.0
				RC[i, j] = 0.0
				RR[i, j] = 0.0
			end
		end
	end

	clims = (0.0, 1.0)
	fig = CairoMakie.Figure(size = (1100, 750))
	grid = CairoMakie.GridLayout()
	fig[1, 1] = grid

	axS = CairoMakie.Axis3(grid[1, 1], title = "S/(S+I+C+R)", xlabel = "t", ylabel = "k", zlabel = "share")
	axI = CairoMakie.Axis3(grid[1, 2], title = "I/(S+I+C+R)", xlabel = "t", ylabel = "k", zlabel = "share")
	axC = CairoMakie.Axis3(grid[2, 1], title = "C/(S+I+C+R)", xlabel = "t", ylabel = "k", zlabel = "share")
	axR = CairoMakie.Axis3(grid[2, 2], title = "R/(S+I+C+R)", xlabel = "t", ylabel = "k", zlabel = "share")

	pltS = _surface_plot!(axS, t, k, RS; colormap = :viridis, colorrange = clims, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)
	_surface_plot!(axI, t, k, RI; colormap = :viridis, colorrange = clims, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)
	_surface_plot!(axC, t, k, RC; colormap = :viridis, colorrange = clims, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)
	_surface_plot!(axR, t, k, RR; colormap = :viridis, colorrange = clims, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)

	CairoMakie.Colorbar(fig[1, 2], pltS)
	outpath = joinpath(outdir, filename)
	if endswith(lowercase(filename), ".jpg") || endswith(lowercase(filename), ".jpeg")
		_save_as_jpg(outpath, fig; px_per_unit = px_per_unit, quality = jpg_quality)
	else
		CairoMakie.save(outpath, fig)
	end
	return nothing
end

function save_figure_3_flux_S_to_I(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_3_heatmap_flux_S_to_I_tk.pdf",
	contour_lines::Int = 6,
	k_indices = nothing,
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
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, FluxSI = _restrict_k(k, kidx, FluxSI)
	flux_hi = maximum(FluxSI)
	clims_flux = _safe_colorrange(0.0, flux_hi)

	fig = CairoMakie.Figure(size = (1100, 650))
	ax = CairoMakie.Axis(fig[1, 1], title = "Flow S→I: β lS*(t,k) ϕS(t,k) LI(t)", xlabel = "t", ylabel = "k")
	hm = _heatmap_with_contours!(ax, t, k, FluxSI; colormap = :viridis, colorrange = clims_flux, contour_lines = contour_lines)
	CairoMakie.Colorbar(fig[1, 2], hm)
	CairoMakie.save(joinpath(outdir, filename), fig)
	return nothing
end

function save_figure_3bis_flux_S_to_I_surface(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_3bis_surface_flux_S_to_I_tk.png",
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
	px_per_unit::Real = 1.35,
	jpg_quality::Int = 92,
	k_indices = nothing,
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
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, FluxSI = _restrict_k(k, kidx, FluxSI)
	flux_hi = maximum(FluxSI)
	clims_flux = _safe_colorrange(0.0, flux_hi)

	fig = CairoMakie.Figure(size = (1050, 650))
	ax = CairoMakie.Axis3(fig[1, 1], title = "Flow S→I", xlabel = "t", ylabel = "k", zlabel = "flow")
	plt = _surface_plot!(ax, t, k, FluxSI; colormap = :viridis, colorrange = clims_flux, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)
	CairoMakie.Colorbar(fig[1, 2], plt)
	outpath = joinpath(outdir, filename)
	if endswith(lowercase(filename), ".jpg") || endswith(lowercase(filename), ".jpeg")
		_save_as_jpg(outpath, fig; px_per_unit = px_per_unit, quality = jpg_quality)
	else
		CairoMakie.save(outpath, fig)
	end
	return nothing
end

function save_figure_4_consumption(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_4_heatmaps_consumption_SICR_tk.pdf",
	contour_lines::Int = 6,
	k_indices = nothing,
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
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, CS, CI, CC, CR = _restrict_k(k, kidx, CS, CI, CC, CR)

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

function save_figure_4bis_consumption_surface(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_4bis_surfaces_consumption_SICR_tk.png",
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
	px_per_unit::Real = 1.35,
	jpg_quality::Int = 92,
	k_indices = nothing,
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
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, CS, CI, CC, CR = _restrict_k(k, kidx, CS, CI, CC, CR)
	c_lo, c_hi = _global_minmax(CS, CI, CC, CR)
	clims_c = _safe_colorrange(c_lo, c_hi)

	fig = CairoMakie.Figure(size = (1100, 750))
	grid = CairoMakie.GridLayout()
	fig[1, 1] = grid

	axS = CairoMakie.Axis3(grid[1, 1], title = "cS(t,k)", xlabel = "t", ylabel = "k", zlabel = "c")
	axI = CairoMakie.Axis3(grid[1, 2], title = "cI(t,k)", xlabel = "t", ylabel = "k", zlabel = "c")
	axC = CairoMakie.Axis3(grid[2, 1], title = "cC(t,k)", xlabel = "t", ylabel = "k", zlabel = "c")
	axR = CairoMakie.Axis3(grid[2, 2], title = "cR(t,k)", xlabel = "t", ylabel = "k", zlabel = "c")

	pltS = _surface_plot!(axS, t, k, CS; colormap = :plasma, colorrange = clims_c, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)
	_surface_plot!(axI, t, k, CI; colormap = :plasma, colorrange = clims_c, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)
	_surface_plot!(axC, t, k, CC; colormap = :plasma, colorrange = clims_c, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)
	_surface_plot!(axR, t, k, CR; colormap = :plasma, colorrange = clims_c, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)

	CairoMakie.Colorbar(fig[1, 2], pltS)
	outpath = joinpath(outdir, filename)
	if endswith(lowercase(filename), ".jpg") || endswith(lowercase(filename), ".jpeg")
		_save_as_jpg(outpath, fig; px_per_unit = px_per_unit, quality = jpg_quality)
	else
		CairoMakie.save(outpath, fig)
	end
	return nothing
end

function save_figure_5_labor(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_5_heatmaps_labor_SIR_tk.pdf",
	contour_lines::Int = 6,
	k_indices = nothing,
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
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, LS, LI, LR = _restrict_k(k, kidx, LS, LI, LR)
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

function save_figure_5bis_labor_surface(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_5bis_surfaces_labor_SIR_tk.png",
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
	px_per_unit::Real = 1.35,
	jpg_quality::Int = 92,
	k_indices = nothing,
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
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, LS, LI, LR = _restrict_k(k, kidx, LS, LI, LR)
	clims_l = (0.0, 1.0)

	fig = CairoMakie.Figure(size = (1200, 550))
	grid = CairoMakie.GridLayout()
	fig[1, 1] = grid

	axS = CairoMakie.Axis3(grid[1, 1], title = "lS(t,k)", xlabel = "t", ylabel = "k", zlabel = "l")
	axI = CairoMakie.Axis3(grid[1, 2], title = "lI(t,k)", xlabel = "t", ylabel = "k", zlabel = "l")
	axR = CairoMakie.Axis3(grid[1, 3], title = "lR(t,k)", xlabel = "t", ylabel = "k", zlabel = "l")

	pltS = _surface_plot!(axS, t, k, LS; colormap = :viridis, colorrange = clims_l, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)
	_surface_plot!(axI, t, k, LI; colormap = :viridis, colorrange = clims_l, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)
	_surface_plot!(axR, t, k, LR; colormap = :viridis, colorrange = clims_l, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)

	CairoMakie.Colorbar(fig[1, 2], pltS)
	outpath = joinpath(outdir, filename)
	if endswith(lowercase(filename), ".jpg") || endswith(lowercase(filename), ".jpeg")
		_save_as_jpg(outpath, fig; px_per_unit = px_per_unit, quality = jpg_quality)
	else
		CairoMakie.save(outpath, fig)
	end
	return nothing
end

function save_figure_6_vaccination_q(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_6_heatmap_q_tk.pdf",
	contour_lines::Int = 6,
	k_indices = nothing,
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
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, Q = _restrict_k(k, kidx, Q)
	q_hi = maximum(Q)
	clims_q = _safe_colorrange(0.0, q_hi)

	fig = CairoMakie.Figure(size = (1100, 650))
	ax = CairoMakie.Axis(fig[1, 1], title = "Vaccination intensity q(t,k)", xlabel = "t", ylabel = "k")
	hm = _heatmap_with_contours!(ax, t, k, Q; colormap = :magma, colorrange = clims_q, contour_lines = contour_lines)
	CairoMakie.Colorbar(fig[1, 2], hm)
	CairoMakie.save(joinpath(outdir, filename), fig)
	return nothing
end

function save_figure_6bis_vaccination_q_surface(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_6bis_surface_q_tk.png",
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
	px_per_unit::Real = 1.35,
	jpg_quality::Int = 92,
	k_indices = nothing,
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
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, Q = _restrict_k(k, kidx, Q)
	q_hi = maximum(Q)
	clims_q = _safe_colorrange(0.0, q_hi)

	fig = CairoMakie.Figure(size = (1050, 650))
	ax = CairoMakie.Axis3(fig[1, 1], title = "Vaccination intensity q(t,k)", xlabel = "t", ylabel = "k", zlabel = "q")
	plt = _surface_plot!(ax, t, k, Q; colormap = :magma, colorrange = clims_q, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)
	CairoMakie.Colorbar(fig[1, 2], plt)
	outpath = joinpath(outdir, filename)
	if endswith(lowercase(filename), ".jpg") || endswith(lowercase(filename), ".jpeg")
		_save_as_jpg(outpath, fig; px_per_unit = px_per_unit, quality = jpg_quality)
	else
		CairoMakie.save(outpath, fig)
	end
	return nothing
end

function save_figure_7_vaccination_flow_S_to_R(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_7_heatmap_flow_S_to_R_vaccination_tk.pdf",
	contour_lines::Int = 6,
	k_indices = nothing,
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

	FlowSR_vax = _time_by_k_matrix(n -> (result.controls[n].q_rate .* result.F[n].ϕSt), Nt, Nk)
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, FlowSR_vax = _restrict_k(k, kidx, FlowSR_vax)
	flow_hi = maximum(FlowSR_vax)
	clims_flow = _safe_colorrange(0.0, flow_hi)

	fig = CairoMakie.Figure(size = (1100, 650))
	ax = CairoMakie.Axis(fig[1, 1], title = "Flow S→R via vaccination: q(t,k) ϕS(t,k)", xlabel = "t", ylabel = "k")
	hm = _heatmap_with_contours!(ax, t, k, FlowSR_vax; colormap = :viridis, colorrange = clims_flow, contour_lines = contour_lines)
	CairoMakie.Colorbar(fig[1, 2], hm)
	CairoMakie.save(joinpath(outdir, filename), fig)
	return nothing
end

function save_figure_7bis_vaccination_flow_S_to_R_surface(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_7bis_surface_flow_S_to_R_vaccination_tk.png",
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
	px_per_unit::Real = 1.35,
	jpg_quality::Int = 92,
	k_indices = nothing,
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

	FlowSR_vax = _time_by_k_matrix(n -> (result.controls[n].q_rate .* result.F[n].ϕSt), Nt, Nk)
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, FlowSR_vax = _restrict_k(k, kidx, FlowSR_vax)
	flow_hi = maximum(FlowSR_vax)
	clims_flow = _safe_colorrange(0.0, flow_hi)

	fig = CairoMakie.Figure(size = (1050, 650))
	ax = CairoMakie.Axis3(fig[1, 1], title = "Flow S→R via vaccination", xlabel = "t", ylabel = "k", zlabel = "flow")
	plt = _surface_plot!(ax, t, k, FlowSR_vax; colormap = :viridis, colorrange = clims_flow, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)
	CairoMakie.Colorbar(fig[1, 2], plt)
	outpath = joinpath(outdir, filename)
	if endswith(lowercase(filename), ".jpg") || endswith(lowercase(filename), ".jpeg")
		_save_as_jpg(outpath, fig; px_per_unit = px_per_unit, quality = jpg_quality)
	else
		CairoMakie.save(outpath, fig)
	end
	return nothing
end

function save_figure_8_effective_wage_S(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_8_heatmap_effective_wage_WS_tk.pdf",
	contour_lines::Int = 6,
	k_indices = nothing,
)
	mkpath(outdir)

	t = result.t
	Nt = length(t)
	Nk = p.Nk
	k = collect(p.k)
	if Nt == 0
		error("result.t is empty")
	end
	if length(result.F) != Nt || length(result.V) != Nt
		error("Inconsistent result lengths: length(t)=$Nt, length(F)=$(length(result.F)), length(V)=$(length(result.V))")
	end

	WS = Matrix{Float64}(undef, Nt, Nk)
	dcache = DerivDkCache(Float64, Nk)
	@inbounds for n in 1:Nt
		Ft = result.F[n]
		Vn = result.V[n]
		Vt = (VS = Vn.VS, VI = Vn.VI, VC = Vn.VC, VR = Vn.VR)
		∂V = compute_∂V_dk!(dcache, Vt, p)
		agg = compute_labor_and_aggregates(Vt, ∂V, Ft, p; w = Vn.w)
		WS[n, :] .= agg.W.WS
	end
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, WS = _restrict_k(k, kidx, WS)

	ws_lo, ws_hi = _finite_minmax(WS)
	clims_ws = _safe_colorrange(ws_lo, ws_hi)

	fig = CairoMakie.Figure(size = (1100, 650))
	ax = CairoMakie.Axis(fig[1, 1], title = "Effective wage for S: W_S(t,k)", xlabel = "t", ylabel = "k")
	hm = _heatmap_with_contours!(ax, t, k, WS; colormap = :viridis, colorrange = clims_ws, contour_lines = contour_lines)
	CairoMakie.Colorbar(fig[1, 2], hm)
	CairoMakie.save(joinpath(outdir, filename), fig)
	return nothing
end

function save_figure_8bis_effective_wage_S_surface(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_8bis_surface_effective_wage_WS_tk.png",
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
	px_per_unit::Real = 1.35,
	jpg_quality::Int = 92,
	k_indices = nothing,
)
	mkpath(outdir)

	t = result.t
	Nt = length(t)
	Nk = p.Nk
	k = collect(p.k)
	if Nt == 0
		error("result.t is empty")
	end
	if length(result.F) != Nt || length(result.V) != Nt
		error("Inconsistent result lengths: length(t)=$Nt, length(F)=$(length(result.F)), length(V)=$(length(result.V))")
	end

	WS = Matrix{Float64}(undef, Nt, Nk)
	dcache = DerivDkCache(Float64, Nk)
	@inbounds for n in 1:Nt
		Ft = result.F[n]
		Vn = result.V[n]
		Vt = (VS = Vn.VS, VI = Vn.VI, VC = Vn.VC, VR = Vn.VR)
		∂V = compute_∂V_dk!(dcache, Vt, p)
		agg = compute_labor_and_aggregates(Vt, ∂V, Ft, p; w = Vn.w)
		WS[n, :] .= agg.W.WS
	end
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, WS = _restrict_k(k, kidx, WS)

	ws_lo, ws_hi = _finite_minmax(WS)
	clims_ws = _safe_colorrange(ws_lo, ws_hi)

	fig = CairoMakie.Figure(size = (1050, 650))
	ax = CairoMakie.Axis3(fig[1, 1], title = "Effective wage for S", xlabel = "t", ylabel = "k", zlabel = "W_S")
	plt = _surface_plot!(ax, t, k, WS; colormap = :viridis, colorrange = clims_ws, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)
	CairoMakie.Colorbar(fig[1, 2], plt)
	outpath = joinpath(outdir, filename)
	if endswith(lowercase(filename), ".jpg") || endswith(lowercase(filename), ".jpeg")
		_save_as_jpg(outpath, fig; px_per_unit = px_per_unit, quality = jpg_quality)
	else
		CairoMakie.save(outpath, fig)
	end
	return nothing
end

function save_figure_9_R0_over_time(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_9_R0_over_time.pdf",
)
	mkpath(outdir)

	t = result.t
	Nt = length(t)
	Nk = p.Nk
	if Nt == 0
		error("result.t is empty")
	end
	if length(result.F) != Nt || length(result.V) != Nt
		error("Inconsistent result lengths: length(t)=$Nt, length(F)=$(length(result.F)), length(V)=$(length(result.V))")
	end

	R0t = fill(NaN, Nt)
	dcache = DerivDkCache(Float64, Nk)
	exitI = p.σ1 + p.σ3 + p.μ
	if !(isfinite(exitI) && exitI > 0)
		error("Non-finite or non-positive exit rate from I: σ1+σ3+μ = $exitI")
	end

	@inbounds for n in 1:Nt
		Ft = result.F[n]
		Vn = result.V[n]
		Vt = (VS = Vn.VS, VI = Vn.VI, VC = Vn.VC, VR = Vn.VR)
		∂V = compute_∂V_dk!(dcache, Vt, p)
		lOpt, _ = optimal_labor_ALL(Vt, ∂V, Ft, Vn.w, p)

		# Integrate along k
		LS = sum(lOpt.lS .* Ft.ϕSt) * p.Δk
		I_mass = sum(Ft.ϕIt) * p.Δk
		LI = sum(lOpt.lI .* Ft.ϕIt) * p.Δk
		lI_bar = I_mass > 0 ? (LI / I_mass) : 0.0

		# Effective reproduction number based on linearized I dynamics:
		# new infections per infected ≈ β * (average infected labor) * (susceptible labor mass)
		R0t[n] = (p.β * lI_bar * LS) / exitI
	end

	fig = CairoMakie.Figure(size = (1000, 500))
	ax = CairoMakie.Axis(fig[1, 1], title = "R₀(t) integrated over k", xlabel = "t", ylabel = "R₀")
	CairoMakie.lines!(ax, t, R0t; linewidth = 2)
	CairoMakie.hlines!(ax, [1.0]; color = (:black, 0.35), linestyle = :dash)
	CairoMakie.save(joinpath(outdir, filename), fig)
	return nothing
end

function save_figure_10_wealth_distribution_total(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_10_heatmap_wealth_distribution_total_tk.pdf",
	contour_lines::Int = 6,
	k_indices = nothing,
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

	Φ = Matrix{Float64}(undef, Nt, Nk)
	@inbounds for n in 1:Nt
		Ft = result.F[n]
		Φ[n, :] .= Ft.ϕSt .+ Ft.ϕIt .+ Ft.ϕCt .+ Ft.ϕRt
	end
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, Φ = _restrict_k(k, kidx, Φ)

	ϕ_lo, ϕ_hi = _finite_minmax(Φ)
	clims_ϕ = _safe_colorrange(ϕ_lo, ϕ_hi)

	fig = CairoMakie.Figure(size = (1100, 650))
	ax = CairoMakie.Axis(fig[1, 1], title = "Total wealth distribution: ϕS+ϕI+ϕC+ϕR", xlabel = "t", ylabel = "k")
	hm = _heatmap_with_contours!(ax, t, k, Φ; colormap = :viridis, colorrange = clims_ϕ, contour_lines = contour_lines)
	CairoMakie.Colorbar(fig[1, 2], hm)
	CairoMakie.save(joinpath(outdir, filename), fig)
	return nothing
end

function save_figure_10bis_wealth_distribution_total_surface(result, p;
	outdir::AbstractString = "outputs/figures",
	filename::AbstractString = "figure_10bis_surface_wealth_distribution_total_tk.png",
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
	px_per_unit::Real = 1.35,
	jpg_quality::Int = 92,
	k_indices = nothing,
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

	Φ = Matrix{Float64}(undef, Nt, Nk)
	@inbounds for n in 1:Nt
		Ft = result.F[n]
		Φ[n, :] .= Ft.ϕSt .+ Ft.ϕIt .+ Ft.ϕCt .+ Ft.ϕRt
	end
	kidx = isnothing(k_indices) ? collect(1:Nk) : k_indices
	k, Φ = _restrict_k(k, kidx, Φ)

	ϕ_lo, ϕ_hi = _finite_minmax(Φ)
	clims_ϕ = _safe_colorrange(ϕ_lo, ϕ_hi)

	fig = CairoMakie.Figure(size = (1050, 650))
	ax = CairoMakie.Axis3(fig[1, 1], title = "Total wealth distribution", xlabel = "t", ylabel = "k", zlabel = "ϕ")
	plt = _surface_plot!(ax, t, k, Φ; colormap = :viridis, colorrange = clims_ϕ, maxNt = maxNt, maxNk = maxNk, rasterize = rasterize)
	CairoMakie.Colorbar(fig[1, 2], plt)
	outpath = joinpath(outdir, filename)
	if endswith(lowercase(filename), ".jpg") || endswith(lowercase(filename), ".jpeg")
		_save_as_jpg(outpath, fig; px_per_unit = px_per_unit, quality = jpg_quality)
	else
		CairoMakie.save(outpath, fig)
	end
	return nothing
end

"""
	save_all_figures(result, p; outdir="outputs/figures", contour_lines=6)

Generate and save a set of figures from the output of `solveModel`.

The main figures are saved as PDF. The "bis" surface figures are saved as PNG by
default.

Each figure is produced by a separate `save_figure_*` function.

Output filenames use the `figure_#_...` prefix (e.g. `figure_1_...`).
Heatmaps are grouped into multi-panel figures (2×2 or 1×3) and include a few
contour lines overlaid to improve readability.

By default, figures with a capital-grid axis are shown only up to the smallest
capital level containing `p.plotKMassLevel` of the initial population mass.
Set `truncate_k_to_initial_mass=false` to plot the full capital grid.
"""
function save_all_figures(result, p;
	outdir::AbstractString = "outputs/figures",
	contour_lines::Int = 6,
	with_surfaces::Bool = true,
	progress::Bool = true,
	truncate_k_to_initial_mass::Bool = p.truncateKPlots,
	k_mass_level::Real = p.plotKMassLevel,
)
	mkpath(outdir)
	k_indices = _plot_k_indices(
		result,
		p;
		truncate_k_to_initial_mass = truncate_k_to_initial_mass,
		k_mass_level = k_mass_level,
	)

	_total = with_surfaces ? 20 : 11
	_pbar = progress ? ProgressMeter.Progress(_total; desc = "Saving figures") : nothing
	_tick(label::AbstractString) = (_pbar === nothing ? nothing : ProgressMeter.next!(_pbar; showvalues = [("step", label)]))

	save_figure_1_totals(result, p; outdir = outdir)
	_tick("figure 1")
	save_figure_2_distributions(result, p; outdir = outdir, contour_lines = contour_lines, k_indices = k_indices)
	_tick("figure 2")
	save_figure_2Rel_relative_shares(result, p; outdir = outdir, contour_lines = contour_lines, k_indices = k_indices)
	_tick("figure 2Rel")
	save_figure_3_flux_S_to_I(result, p; outdir = outdir, contour_lines = contour_lines, k_indices = k_indices)
	_tick("figure 3")
	save_figure_4_consumption(result, p; outdir = outdir, contour_lines = contour_lines, k_indices = k_indices)
	_tick("figure 4")
	save_figure_5_labor(result, p; outdir = outdir, contour_lines = contour_lines, k_indices = k_indices)
	_tick("figure 5")
	save_figure_6_vaccination_q(result, p; outdir = outdir, contour_lines = contour_lines, k_indices = k_indices)
	_tick("figure 6")
	save_figure_7_vaccination_flow_S_to_R(result, p; outdir = outdir, contour_lines = contour_lines, k_indices = k_indices)
	_tick("figure 7")
	save_figure_8_effective_wage_S(result, p; outdir = outdir, contour_lines = contour_lines, k_indices = k_indices)
	_tick("figure 8")
	save_figure_9_R0_over_time(result, p; outdir = outdir)
	_tick("figure 9")
	save_figure_10_wealth_distribution_total(result, p; outdir = outdir, contour_lines = contour_lines, k_indices = k_indices)
	_tick("figure 10")

	if with_surfaces
		save_figure_2bis_distributions_surface(result, p; outdir = outdir, k_indices = k_indices)
		_tick("figure 2bis")
		save_figure_2Relbis_relative_shares_surface(result, p; outdir = outdir, k_indices = k_indices)
		_tick("figure 2Relbis")
		save_figure_3bis_flux_S_to_I_surface(result, p; outdir = outdir, k_indices = k_indices)
		_tick("figure 3bis")
		save_figure_4bis_consumption_surface(result, p; outdir = outdir, k_indices = k_indices)
		_tick("figure 4bis")
		save_figure_5bis_labor_surface(result, p; outdir = outdir, k_indices = k_indices)
		_tick("figure 5bis")
		save_figure_6bis_vaccination_q_surface(result, p; outdir = outdir, k_indices = k_indices)
		_tick("figure 6bis")
		save_figure_7bis_vaccination_flow_S_to_R_surface(result, p; outdir = outdir, k_indices = k_indices)
		_tick("figure 7bis")
		save_figure_8bis_effective_wage_S_surface(result, p; outdir = outdir, k_indices = k_indices)
		_tick("figure 8bis")
		save_figure_10bis_wealth_distribution_total_surface(result, p; outdir = outdir, k_indices = k_indices)
		_tick("figure 10bis")
	end

	return nothing
end

function _csv_write_row(io, xs...)
	println(io, join((string(x) for x in xs), ","))
	return nothing
end

function _maybe_property(x, name::Symbol, default = NaN)
	return hasproperty(x, name) ? getproperty(x, name) : default
end

function _maybe_vector_value(x, name::Symbol, n::Int, default = NaN)
	if hasproperty(x, name)
		v = getproperty(x, name)
		return v[n]
	end
	return default
end

"""
	save_solution_csv(result, p; outdir="outputs/solution_csv")

Save the numerical solution in long-form CSV files. The exported files are meant
to make it possible to redraw figures later without re-running the model.
"""
function save_solution_csv(result, p;
	outdir::AbstractString = "outputs/solution_csv",
)
	mkpath(outdir)

	t = result.t
	Nt = length(t)
	Nk = p.Nk
	k = collect(p.k)

	open(joinpath(outdir, "metadata.csv"), "w") do io
		_csv_write_row(io, "key", "value")
		_csv_write_row(io, "Nk", p.Nk)
		_csv_write_row(io, "MaxK", p.MaxK)
		_csv_write_row(io, "DeltaK", p.Δk)
		_csv_write_row(io, "T_End", p.T_End)
		_csv_write_row(io, "DeltaT", p.Δt)
		_csv_write_row(io, "xi", p.ξ)
		_csv_write_row(io, "method", _maybe_property(result, :method, "quasistatic"))
		_csv_write_row(io, "converged", _maybe_property(result, :converged, ""))
		_csv_write_row(io, "iterations", _maybe_property(result, :iterations, ""))
	end

	if haskey(result, :F) && length(result.F) == Nt
		open(joinpath(outdir, "distributions.csv"), "w") do io
			_csv_write_row(io, "time_index", "t", "k_index", "k", "phiS", "phiI", "phiC", "phiR")
			@inbounds for n in 1:Nt
				Ft = result.F[n]
				for i in 1:Nk
					_csv_write_row(io, n, t[n], i, k[i], Ft.ϕSt[i], Ft.ϕIt[i], Ft.ϕCt[i], Ft.ϕRt[i])
				end
			end
		end
	end

	if haskey(result, :V) && length(result.V) == Nt
		open(joinpath(outdir, "values.csv"), "w") do io
			_csv_write_row(io, "time_index", "t", "k_index", "k", "VS", "VI", "VC", "VR", "w", "r", "LI")
			@inbounds for n in 1:Nt
				Vn = result.V[n]
				w = _maybe_property(Vn, :w)
				r = _maybe_property(Vn, :r)
				LI = _maybe_property(Vn, :LI)
				for i in 1:Nk
					_csv_write_row(io, n, t[n], i, k[i], Vn.VS[i], Vn.VI[i], Vn.VC[i], Vn.VR[i], w, r, LI)
				end
			end
		end
	end

	if haskey(result, :controls) && length(result.controls) == Nt
		open(joinpath(outdir, "controls.csv"), "w") do io
			_csv_write_row(
				io,
				"time_index", "t", "k_index", "k",
				"cS", "cI", "cC", "cR",
				"lS", "lI", "lC", "lR",
				"q", "xiS",
				"bS", "bI", "bC", "bR",
				"infection_rate",
			)
			@inbounds for n in 1:Nt
				c = result.controls[n]
				xiS = _maybe_property(c, :ξS, fill(NaN, Nk))
				for i in 1:Nk
					_csv_write_row(
						io,
						n, t[n], i, k[i],
						c.cS[i], c.cI[i], c.cC[i], c.cR[i],
						c.lOpt.lS[i], c.lOpt.lI[i], c.lOpt.lC[i], c.lOpt.lR[i],
						c.q_rate[i], xiS[i],
						c.bS[i], c.bI[i], c.bC[i], c.bR[i],
						c.infection_rate[i],
					)
				end
			end
		end

		open(joinpath(outdir, "aggregates_prices.csv"), "w") do io
			_csv_write_row(io, "time_index", "t", "K", "L", "LI", "w", "r")
			prices = _maybe_property(result, :prices, nothing)
			@inbounds for n in 1:Nt
				c = result.controls[n]
				K = prices === nothing ? _maybe_property(c, :K) : _maybe_vector_value(prices, :K, n)
				L = prices === nothing ? _maybe_property(c, :L) : _maybe_vector_value(prices, :L, n)
				LI = prices === nothing ? _maybe_property(c, :LI) : _maybe_vector_value(prices, :LI, n)
				w = prices === nothing ? _maybe_property(c, :w) : _maybe_vector_value(prices, :w, n)
				r = prices === nothing ? _maybe_property(c, :r) : _maybe_vector_value(prices, :r, n)
				_csv_write_row(io, n, t[n], K, L, LI, w, r)
			end
		end
	end

	if haskey(result, :diagnostics)
		diag = result.diagnostics
		if hasproperty(diag, :err)
			open(joinpath(outdir, "dynamic_diagnostics.csv"), "w") do io
				_csv_write_row(
					io,
					"iteration", "err", "errF", "errV", "errW", "errR",
					"mass_error", "min_density", "max_negative_before_projection",
					"normalization_correction",
				)
				for it in eachindex(diag.err)
					_csv_write_row(
						io,
						it,
						diag.err[it],
						diag.errF[it],
						diag.errV[it],
						diag.errW[it],
						diag.errR[it],
						diag.mass_error[it],
						diag.min_density[it],
						diag.max_negative_before_projection[it],
						diag.normalization_correction[it],
					)
				end
			end
		end
	end

	return outdir
end

function _read_csv_rows(path::AbstractString)
	lines = readlines(path)
	if isempty(lines)
		error("CSV file is empty: $path")
	end
	header = split(lines[1], ",")
	rows = [split(line, ",") for line in lines[2:end] if !isempty(strip(line))]
	return header, rows
end

function _csv_index(header, name::AbstractString)
	idx = findfirst(==(name), header)
	if idx === nothing
		error("Missing CSV column '$name'")
	end
	return idx
end

function _parse_csv_float(x)
	s = strip(String(x))
	return isempty(s) ? NaN : parse(Float64, s)
end

function _parse_csv_int(x)
	return parse(Int, strip(String(x)))
end

function _load_distributions_csv(path::AbstractString, p)
	header, rows = _read_csv_rows(path)
	it = _csv_index(header, "time_index")
	tt = _csv_index(header, "t")
	ik = _csv_index(header, "k_index")
	iS = _csv_index(header, "phiS")
	iI = _csv_index(header, "phiI")
	iC = _csv_index(header, "phiC")
	iR = _csv_index(header, "phiR")

	Nt = maximum(_parse_csv_int(row[it]) for row in rows)
	t = zeros(Float64, Nt)
	ΦS = [zeros(Float64, p.Nk) for _ in 1:Nt]
	ΦI = [zeros(Float64, p.Nk) for _ in 1:Nt]
	ΦC = [zeros(Float64, p.Nk) for _ in 1:Nt]
	ΦR = [zeros(Float64, p.Nk) for _ in 1:Nt]

	for row in rows
		n = _parse_csv_int(row[it])
		i = _parse_csv_int(row[ik])
		t[n] = _parse_csv_float(row[tt])
		ΦS[n][i] = _parse_csv_float(row[iS])
		ΦI[n][i] = _parse_csv_float(row[iI])
		ΦC[n][i] = _parse_csv_float(row[iC])
		ΦR[n][i] = _parse_csv_float(row[iR])
	end

	F = [(ϕSt = ΦS[n], ϕIt = ΦI[n], ϕCt = ΦC[n], ϕRt = ΦR[n]) for n in 1:Nt]
	return t, F
end

function _load_values_csv(path::AbstractString, p, Nt::Int)
	header, rows = _read_csv_rows(path)
	it = _csv_index(header, "time_index")
	ik = _csv_index(header, "k_index")
	iVS = _csv_index(header, "VS")
	iVI = _csv_index(header, "VI")
	iVC = _csv_index(header, "VC")
	iVR = _csv_index(header, "VR")
	iw = _csv_index(header, "w")
	ir = _csv_index(header, "r")
	iLI = _csv_index(header, "LI")

	VS = [zeros(Float64, p.Nk) for _ in 1:Nt]
	VI = [zeros(Float64, p.Nk) for _ in 1:Nt]
	VC = [zeros(Float64, p.Nk) for _ in 1:Nt]
	VR = [zeros(Float64, p.Nk) for _ in 1:Nt]
	w = fill(NaN, Nt)
	r = fill(NaN, Nt)
	LI = fill(NaN, Nt)

	for row in rows
		n = _parse_csv_int(row[it])
		i = _parse_csv_int(row[ik])
		VS[n][i] = _parse_csv_float(row[iVS])
		VI[n][i] = _parse_csv_float(row[iVI])
		VC[n][i] = _parse_csv_float(row[iVC])
		VR[n][i] = _parse_csv_float(row[iVR])
		if i == 1
			w[n] = _parse_csv_float(row[iw])
			r[n] = _parse_csv_float(row[ir])
			LI[n] = _parse_csv_float(row[iLI])
		end
	end

	return [(VS = VS[n], VI = VI[n], VC = VC[n], VR = VR[n], w = w[n], r = r[n], LI = LI[n]) for n in 1:Nt]
end

function _load_controls_csv(path::AbstractString, p, Nt::Int)
	header, rows = _read_csv_rows(path)
	it = _csv_index(header, "time_index")
	ik = _csv_index(header, "k_index")
	cols = Dict(name => _csv_index(header, name) for name in (
		"cS", "cI", "cC", "cR",
		"lS", "lI", "lC", "lR",
		"q", "xiS",
		"bS", "bI", "bC", "bR",
		"infection_rate",
	))

	data = Dict(name => [zeros(Float64, p.Nk) for _ in 1:Nt] for name in keys(cols))
	for row in rows
		n = _parse_csv_int(row[it])
		i = _parse_csv_int(row[ik])
		for (name, col) in cols
			data[name][n][i] = _parse_csv_float(row[col])
		end
	end

	return [
		(
			lOpt = (lS = data["lS"][n], lI = data["lI"][n], lC = data["lC"][n], lR = data["lR"][n]),
			cS = data["cS"][n],
			cI = data["cI"][n],
			cC = data["cC"][n],
			cR = data["cR"][n],
			bS = data["bS"][n],
			bI = data["bI"][n],
			bC = data["bC"][n],
			bR = data["bR"][n],
			infection_rate = data["infection_rate"][n],
			q_rate = data["q"][n],
			ξS = data["xiS"][n],
		)
		for n in 1:Nt
	]
end

function _load_aggregates_prices_csv(path::AbstractString, Nt::Int)
	header, rows = _read_csv_rows(path)
	it = _csv_index(header, "time_index")
	iK = _csv_index(header, "K")
	iL = _csv_index(header, "L")
	iLI = _csv_index(header, "LI")
	iw = _csv_index(header, "w")
	ir = _csv_index(header, "r")

	K = fill(NaN, Nt)
	L = fill(NaN, Nt)
	LI = fill(NaN, Nt)
	w = fill(NaN, Nt)
	r = fill(NaN, Nt)
	for row in rows
		n = _parse_csv_int(row[it])
		K[n] = _parse_csv_float(row[iK])
		L[n] = _parse_csv_float(row[iL])
		LI[n] = _parse_csv_float(row[iLI])
		w[n] = _parse_csv_float(row[iw])
		r[n] = _parse_csv_float(row[ir])
	end
	return (prices = (w = w, r = r, K = K, L = L, LI = LI), aggregates = (K = K, L = L, LI = LI))
end

function _load_dynamic_diagnostics_csv(path::AbstractString)
	header, rows = _read_csv_rows(path)
	err = Float64[]
	errF = Float64[]
	errV = Float64[]
	errW = Float64[]
	errR = Float64[]
	mass_error = Float64[]
	min_density = Float64[]
	max_negative = Float64[]
	normalization_correction = Float64[]

	for row in rows
		push!(err, _parse_csv_float(row[_csv_index(header, "err")]))
		push!(errF, _parse_csv_float(row[_csv_index(header, "errF")]))
		push!(errV, _parse_csv_float(row[_csv_index(header, "errV")]))
		push!(errW, _parse_csv_float(row[_csv_index(header, "errW")]))
		push!(errR, _parse_csv_float(row[_csv_index(header, "errR")]))
		push!(mass_error, _parse_csv_float(row[_csv_index(header, "mass_error")]))
		push!(min_density, _parse_csv_float(row[_csv_index(header, "min_density")]))
		push!(max_negative, _parse_csv_float(row[_csv_index(header, "max_negative_before_projection")]))
		push!(normalization_correction, _parse_csv_float(row[_csv_index(header, "normalization_correction")]))
	end

	return (
		err = err,
		errF = errF,
		errV = errV,
		errW = errW,
		errR = errR,
		mass_error = mass_error,
		min_density = min_density,
		max_negative_before_projection = max_negative,
		normalization_correction = normalization_correction,
	)
end

"""
	load_solution_csv(outdir, p)

Reconstruct a result-like object from CSV files produced by `save_solution_csv`.
Pass a parameter object `p` with the same capital grid used to generate the CSVs.
"""
function load_solution_csv(outdir::AbstractString, p)
	dist_path = joinpath(outdir, "distributions.csv")
	if !isfile(dist_path)
		error("Missing distributions.csv in $outdir")
	end

	t, F = _load_distributions_csv(dist_path, p)
	Nt = length(t)
	V = isfile(joinpath(outdir, "values.csv")) ? _load_values_csv(joinpath(outdir, "values.csv"), p, Nt) : Any[]
	controls = isfile(joinpath(outdir, "controls.csv")) ? _load_controls_csv(joinpath(outdir, "controls.csv"), p, Nt) : Any[]
	ap = isfile(joinpath(outdir, "aggregates_prices.csv")) ?
		_load_aggregates_prices_csv(joinpath(outdir, "aggregates_prices.csv"), Nt) :
		(prices = nothing, aggregates = nothing)
	diag_path = joinpath(outdir, "dynamic_diagnostics.csv")

	if isfile(diag_path)
		return (
			t = t,
			F = F,
			V = V,
			controls = controls,
			prices = ap.prices,
			aggregates = ap.aggregates,
			diagnostics = _load_dynamic_diagnostics_csv(diag_path),
			method = :loaded_from_csv,
		)
	end

	if ap.prices === nothing
		return (t = t, F = F, V = V, controls = controls, method = :loaded_from_csv)
	end

	return (
		t = t,
		F = F,
		V = V,
		controls = controls,
		prices = ap.prices,
		aggregates = ap.aggregates,
		method = :loaded_from_csv,
	)
end
