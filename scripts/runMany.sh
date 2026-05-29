#!/usr/bin/env bash

set -u

cd /home/cricci/juliadev/EpiEconMFG || exit 1

run_quasistatic() {
    local name="$1"
    local params="$2"
    local outdir="outputs/${name}"

    mkdir -p "$outdir"
    julia --project=. -e "include(\"main.jl\"); p = EpiEconMFG.MFGEpiEcon(${params}); result = run(p = p, show_progress = false, outdir = \"${outdir}\"); EpiEconMFG.save_all_figures(result, p; outdir = \"${outdir}\")" \
        > "${outdir}/output.log" 2>&1 &
}
# 
run_dynamic_case() {
    local name="$1"
    local params="$2"
    local outdir="outputs/${name}"

    mkdir -p "$outdir"
    julia --project=. -e "include(\"main.jl\"); p = EpiEconMFG.MFGEpiEcon(${params}); result = run_dynamic(p = p, show_progress = false, outdir = \"${outdir}\"); EpiEconMFG.save_all_figures(result, p; outdir = \"${outdir}\")" \
        > "${outdir}/output.log" 2>&1 &
}

# Quasi-static HJB runs.
# run_quasistatic "quasi-static-HJB_BASELINE" ""
# run_quasistatic "quasi-static-HJB_dIdC12" "dI = 1.0, dC = 2.0"
# run_quasistatic "quasi-static-HJB_dIdC1020" "dI = 10.0, dC = 20.0"
# run_quasistatic "quasi-static-HJB_dIdC12xi001" "dI = 1.0, dC = 2.0, ξ = 0.01"
# run_quasistatic "quasi-static-HJB_dIdC1020xi001" "dI = 10.0, dC = 20.0, ξ = 0.01"
# run_quasistatic "quasi-static-HJB_dIdC12xi01" "dI = 1.0, dC = 2.0, ξ = 0.1"
# run_quasistatic "quasi-static-HJB_dIdC1020xi01" "dI = 10.0, dC = 20.0, ξ = 0.1"
run_quasistatic "quasi-static-BASELINE_long" "T_End = 40.0"

# Example dynamic runs. Uncomment if you want to launch them too.
run_dynamic_case "dynamic_NoEpidemic" "I0 = 0.0"
run_dynamic_case "dynamic_BASELINE" ""
run_dynamic_case "dynamic_BASELINE_long" "T_End = 40.0, maxIterDynamic = 1000000000"




# run_dynamic_case "dynamic_T2_xi0001" "T_End = 2.0, ξ = 0.001"
# run_dynamic_case "dynamic_T3_xi0001" "T_End = 3.0, ξ = 0.001"

wait
echo "All Julia jobs completed."

# ---------------------------------------------------------------------------
# Plot only from already-saved CSVs.
#
# These examples do not re-run the model. They reconstruct a result-like object
# from CSV files saved by run(...; outdir=...) or run_dynamic(...; outdir=...),
# then call save_all_figures.
#
# Important: pass a parameter object with the same grid/calibration used to
# create the CSVs.
#
# julia --project=. -e 'include("main.jl"); p = EpiEconMFG.MFGEpiEcon(dI = 20.0, dC = 40.0, ξ = 0.5); result = EpiEconMFG.load_solution_csv("outputs/dynamic_dIdC2040xi05", p); EpiEconMFG.save_all_figures(result, p; outdir="outputs/dynamic_dIdC2040xi05_redrawn_from_csv")'
# julia --project=. -e 'include("main.jl"); p = EpiEconMFG.MFGEpiEcon(ξ = 0.002); result = EpiEconMFG.load_solution_csv("outputs/quasi-static-HJB_xi0.002", p); EpiEconMFG.save_all_figures(result, p; outdir="outputs/quasi-static-HJB_xi0.002_redrawn_from_csv")'
# julia --project=. -e 'include("main.jl"); p = EpiEconMFG.MFGEpiEcon(T_End = 3.0, ξ = 0.001); result = EpiEconMFG.load_solution_csv("outputs/dynamic_T3_xi0001", p); EpiEconMFG.save_all_figures(result, p; outdir="outputs/dynamic_T3_xi0001_redrawn_from_csv")'
