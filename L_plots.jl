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
	@warn "JPEG not supported; wrote PNG instead" jpg_path = path png_path = png_path
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

function save_figure_2Rel_relative_shares(result, p;
	outdir::AbstractString = "figures",
	filename::AbstractString = "figure_2Rel_heatmaps_relative_shares_SICR_tk.pdf",
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
	outdir::AbstractString = "figures",
	filename::AbstractString = "figure_2bis_surfaces_distributions_SICR_tk.jpg",
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
	px_per_unit::Real = 1.35,
	jpg_quality::Int = 92,
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
	outdir::AbstractString = "figures",
	filename::AbstractString = "figure_2Relbis_surfaces_relative_shares_SICR_tk.jpg",
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
	px_per_unit::Real = 1.35,
	jpg_quality::Int = 92,
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

function save_figure_3bis_flux_S_to_I_surface(result, p;
	outdir::AbstractString = "figures",
	filename::AbstractString = "figure_3bis_surface_flux_S_to_I_tk.jpg",
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
	px_per_unit::Real = 1.35,
	jpg_quality::Int = 92,
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

function save_figure_4bis_consumption_surface(result, p;
	outdir::AbstractString = "figures",
	filename::AbstractString = "figure_4bis_surfaces_consumption_SICR_tk.jpg",
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
	px_per_unit::Real = 1.35,
	jpg_quality::Int = 92,
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

function save_figure_5bis_labor_surface(result, p;
	outdir::AbstractString = "figures",
	filename::AbstractString = "figure_5bis_surfaces_labor_SIR_tk.jpg",
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
	px_per_unit::Real = 1.35,
	jpg_quality::Int = 92,
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

function save_figure_6bis_vaccination_q_surface(result, p;
	outdir::AbstractString = "figures",
	filename::AbstractString = "figure_6bis_surface_q_tk.jpg",
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
	px_per_unit::Real = 1.35,
	jpg_quality::Int = 92,
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
	outdir::AbstractString = "figures",
	filename::AbstractString = "figure_7_heatmap_flow_S_to_R_vaccination_tk.pdf",
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

	FlowSR_vax = _time_by_k_matrix(n -> (result.controls[n].q_rate .* result.F[n].ϕSt), Nt, Nk)
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
	outdir::AbstractString = "figures",
	filename::AbstractString = "figure_7bis_surface_flow_S_to_R_vaccination_tk.jpg",
	maxNt::Int = 140,
	maxNk::Int = 140,
	rasterize = 1,
	px_per_unit::Real = 1.35,
	jpg_quality::Int = 92,
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

"""
	save_all_figures(result, p; outdir="figures", contour_lines=6)

Generate and save a set of figures from the output of `solveModel`.

The main figures are saved as PDF. The "bis" surface figures are saved as JPG by
default (via PNG rendering + conversion on macOS).

Each figure is produced by a separate `save_figure_*` function.

Output filenames use the `figure_#_...` prefix (e.g. `figure_1_...`).
Heatmaps are grouped into multi-panel figures (2×2 or 1×3) and include a few
contour lines overlaid to improve readability.
"""
function save_all_figures(result, p;
	outdir::AbstractString = "figures",
	contour_lines::Int = 6,
	with_surfaces::Bool = true,
	progress::Bool = true,
)
	mkpath(outdir)

	_total = with_surfaces ? 15 : 8
	_pbar = progress ? ProgressMeter.Progress(_total; desc = "Saving figures") : nothing
	_tick(label::AbstractString) = (_pbar === nothing ? nothing : ProgressMeter.next!(_pbar; showvalues = [("step", label)]))

	save_figure_1_totals(result, p; outdir = outdir)
	_tick("figure 1")
	save_figure_2_distributions(result, p; outdir = outdir, contour_lines = contour_lines)
	_tick("figure 2")
	save_figure_2Rel_relative_shares(result, p; outdir = outdir, contour_lines = contour_lines)
	_tick("figure 2Rel")
	save_figure_3_flux_S_to_I(result, p; outdir = outdir, contour_lines = contour_lines)
	_tick("figure 3")
	save_figure_4_consumption(result, p; outdir = outdir, contour_lines = contour_lines)
	_tick("figure 4")
	save_figure_5_labor(result, p; outdir = outdir, contour_lines = contour_lines)
	_tick("figure 5")
	save_figure_6_vaccination_q(result, p; outdir = outdir, contour_lines = contour_lines)
	_tick("figure 6")
	save_figure_7_vaccination_flow_S_to_R(result, p; outdir = outdir, contour_lines = contour_lines)
	_tick("figure 7")

	if with_surfaces
		save_figure_2bis_distributions_surface(result, p; outdir = outdir)
		_tick("figure 2bis")
		save_figure_2Relbis_relative_shares_surface(result, p; outdir = outdir)
		_tick("figure 2Relbis")
		save_figure_3bis_flux_S_to_I_surface(result, p; outdir = outdir)
		_tick("figure 3bis")
		save_figure_4bis_consumption_surface(result, p; outdir = outdir)
		_tick("figure 4bis")
		save_figure_5bis_labor_surface(result, p; outdir = outdir)
		_tick("figure 5bis")
		save_figure_6bis_vaccination_q_surface(result, p; outdir = outdir)
		_tick("figure 6bis")
		save_figure_7bis_vaccination_flow_S_to_R_surface(result, p; outdir = outdir)
		_tick("figure 7bis")
	end

	return nothing
end

