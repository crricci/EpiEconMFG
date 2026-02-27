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

"""
	save_all_figures(result, p; outdir="figures")

Generate and save a set of PDF figures from the output of `solveModel`.

Input
- `result`: the `NamedTuple` returned by `solveModel` (fields `t`, `F`, `V`, `controls`).
- `p`: model parameters (`MFGEpiEcon`), used for the capital grid `p.k`, spacing `p.Δk`, and
  parameters like `p.β` and `p.qMax`.

Output
- Writes PDF files into `outdir`.

Figures saved (as separate PDFs):
1. Totals over time for S/I/C/R (integrated over k).
2. Heatmaps of S(t,k), I(t,k), C(t,k), R(t,k) with a shared colorscale.
3. Heatmap of S→I flow: β * lS*(t,k) * ϕS(t,k) * LI(t).
4. Heatmaps of optimal consumption cS/cI/cC/cR with a shared colorscale.
5. Heatmaps of optimal labor lS/lI/lR with a shared colorscale.
6. Heatmap of vaccination intensity q(t,k).
"""
function save_all_figures(result, p; outdir::AbstractString = "figures")
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

	# ---------- 1) Totals over time (integrate over k)
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
	CairoMakie.save(joinpath(outdir, "01_totals_SICR_over_time.pdf"), fig)

	# Helper to save a single heatmap
	function save_heatmap_pdf(filename::AbstractString, Z_tk;
		title::AbstractString,
		colorrange,
		colormap = :viridis,
		xlabel::AbstractString = "t",
		ylabel::AbstractString = "k")

		f = CairoMakie.Figure(size = (900, 600))
		a = CairoMakie.Axis(f[1, 1], xlabel = xlabel, ylabel = ylabel, title = title)
		hm = CairoMakie.heatmap!(a, t, k, Z_tk; colormap = colormap, colorrange = colorrange)
		CairoMakie.Colorbar(f[1, 2], hm)
		CairoMakie.save(joinpath(outdir, filename), f)
		return nothing
	end

	# ---------- 2) Distributions S/I/C/R heatmaps (shared colorscale)
	ΦS = _time_by_k_matrix(n -> result.F[n].ϕSt, Nt, Nk)
	ΦI = _time_by_k_matrix(n -> result.F[n].ϕIt, Nt, Nk)
	ΦC = _time_by_k_matrix(n -> result.F[n].ϕCt, Nt, Nk)
	ΦR = _time_by_k_matrix(n -> result.F[n].ϕRt, Nt, Nk)

	ϕ_hi = maximum((maximum(ΦS), maximum(ΦI), maximum(ΦC), maximum(ΦR)))
	clims_ϕ = (0.0, ϕ_hi)

	save_heatmap_pdf("02_heatmap_S_tk.pdf", ΦS; title = "S(t,k)", colorrange = clims_ϕ)
	save_heatmap_pdf("03_heatmap_I_tk.pdf", ΦI; title = "I(t,k)", colorrange = clims_ϕ)
	save_heatmap_pdf("04_heatmap_C_tk.pdf", ΦC; title = "C(t,k)", colorrange = clims_ϕ)
	save_heatmap_pdf("05_heatmap_R_tk.pdf", ΦR; title = "R(t,k)", colorrange = clims_ϕ)

	# ---------- 3) Flow S -> I: β * lS*(t,k) * ϕS(t,k) * LI(t)
	FluxSI = _time_by_k_matrix(n -> (result.controls[n].infection_rate .* result.F[n].ϕSt), Nt, Nk)
	flux_hi = maximum(FluxSI)
	save_heatmap_pdf(
		"06_heatmap_flux_S_to_I_tk.pdf",
		FluxSI;
		title = "Flow S→I: β lS*(t,k) ϕS(t,k) LI(t)",
		colorrange = (0.0, flux_hi),
	)

	# ---------- 4) Consumption heatmaps (shared colorscale)
	CS = _time_by_k_matrix(n -> result.controls[n].cS, Nt, Nk)
	CI = _time_by_k_matrix(n -> result.controls[n].cI, Nt, Nk)
	CC = _time_by_k_matrix(n -> result.controls[n].cC, Nt, Nk)
	CR = _time_by_k_matrix(n -> result.controls[n].cR, Nt, Nk)
	c_lo, c_hi = _global_minmax(CS, CI, CC, CR)
	clims_c = (c_lo, c_hi)

	save_heatmap_pdf("07_heatmap_cS_tk.pdf", CS; title = "Consumption cS(t,k)", colorrange = clims_c, colormap = :plasma)
	save_heatmap_pdf("08_heatmap_cI_tk.pdf", CI; title = "Consumption cI(t,k)", colorrange = clims_c, colormap = :plasma)
	save_heatmap_pdf("09_heatmap_cC_tk.pdf", CC; title = "Consumption cC(t,k)", colorrange = clims_c, colormap = :plasma)
	save_heatmap_pdf("10_heatmap_cR_tk.pdf", CR; title = "Consumption cR(t,k)", colorrange = clims_c, colormap = :plasma)

	# ---------- 5) Labor heatmaps (shared colorscale, C is always 0 so omitted)
	LS = _time_by_k_matrix(n -> result.controls[n].lOpt.lS, Nt, Nk)
	LI = _time_by_k_matrix(n -> result.controls[n].lOpt.lI, Nt, Nk)
	LR = _time_by_k_matrix(n -> result.controls[n].lOpt.lR, Nt, Nk)
	clims_l = (0.0, 1.0)

	save_heatmap_pdf("11_heatmap_lS_tk.pdf", LS; title = "Labor lS(t,k)", colorrange = clims_l, colormap = :viridis)
	save_heatmap_pdf("12_heatmap_lI_tk.pdf", LI; title = "Labor lI(t,k)", colorrange = clims_l, colormap = :viridis)
	save_heatmap_pdf("13_heatmap_lR_tk.pdf", LR; title = "Labor lR(t,k)", colorrange = clims_l, colormap = :viridis)

	# ---------- 6) Vaccination intensity q(t,k)
	Q = _time_by_k_matrix(n -> result.controls[n].q_rate, Nt, Nk)
	save_heatmap_pdf(
		"14_heatmap_q_tk.pdf",
		Q;
		title = "Vaccination intensity q(t,k)",
		colorrange = (0.0, p.qMax),
		colormap = :magma,
	)

	return nothing
end

