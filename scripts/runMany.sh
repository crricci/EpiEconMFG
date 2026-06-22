#!/usr/bin/env bash

set -u
cd /home/cricci/juliadev/EpiEconMFG || exit 1

load_telegram_env() {
    if [[ -f "${HOME}/.bashrc" ]]; then
        set -a
        # shellcheck source=/dev/null
        source "${HOME}/.bashrc" >/dev/null 2>&1 || true
        set +a
    fi
}

notify_done() {
    local text="$1"

    load_telegram_env
    if [[ -n "${TELEGRAM_BOT_TOKEN:-}" && -n "${TELEGRAM_CHAT_ID:-}" ]]; then
        python3 /home/cricci/vscode/notify_telegram.py --text "$text" || true
        return
    fi

    # Fallback for .bashrc files that only initialize variables in interactive shells.
    bash -ic 'python3 /home/cricci/vscode/notify_telegram.py --text "$1"' _ "$text" >/dev/null 2>&1 || true
}

run_quasistatic() {
    local name="$1"
    local params="$2"
    local outdir="outputs/${name}"

    mkdir -p "$outdir"
    julialauncher --project=. -e "include(\"main.jl\"); p = EpiEconMFG.MFGEpiEcon(${params}); result = run(p = p, show_progress = false, outdir = \"${outdir}\"); EpiEconMFG.save_all_figures(result, p; outdir = \"${outdir}\")" \
        > "${outdir}/output.log" 2>&1 &
}
# 
run_dynamic_case() {
    local name="$1"
    local params="$2"
    local outdir="outputs/${name}"

    mkdir -p "$outdir"
    julialauncher --project=. -e "include(\"main.jl\"); p = EpiEconMFG.MFGEpiEcon(${params}); result = run_dynamic(p = p, show_progress = false, outdir = \"${outdir}\"); EpiEconMFG.save_all_figures(result, p; outdir = \"${outdir}\")" \
        > "${outdir}/output.log" 2>&1 &
}

run_dynamic_case "dynamic_NoEpidemic" "I0 = 0.0"

# Restart from a CSV initial condition saved in data/initial_conditions.
# The CSV below was copied from the final distribution of
# outputs/dynamic_NoEpidemic_long_restart_from_finalS and requires MaxK=20, Δk=0.2.
# The CSV supplies only the wealth distribution over k; N and I0 still set the
# initial infected share I0/N.
# Use single quotes around the parameter list so the Julia string path stays quoted.
#
# run_dynamic_case "dynamic_from_finalS_csv_T60" 'N = 1e4, I0 = 1.0, useCsvInitialDistribution = true, initialDistributionCsvDir = "data/initial_conditions/dynamic_NoEpidemic_long_restart_from_finalS"'

# run_dynamic_case "dynamic_BASELINE" ""
# run_dynamic_case "dynamic_VACCINEFREE" "ξ = 0.0"
# run_dynamic_case "dynamic_NOVACCINE" "qMax = 0.0"
# run_dynamic_case "dynamic_VACCINE_LINEAR_K_0_025" "vaccineCostProfile = :linear_k, ξKMin = 0.0, ξKMax = 0.25"


run_dynamic_case "dynamic_Beta400" "β = 400.0"
run_dynamic_case "dynamic_Beta600" "β = 600.0"
run_dynamic_case "dynamic_Sigma1_14DaysSigma3_14Days" "σ1 = 365/14, σ3 = 365/14"
run_dynamic_case "dynamic_Sigma1_21DaysSigma3_21Days" "σ1 = 365/21, σ3 = 365/21"

run_dynamic_case "dynamic_Beta300Sigma1_14Days_BASELINE" "β = 300.0, σ1 = 365/14"
run_dynamic_case "dynamic_Beta300Sigma1_14Days_VACCINEFREE" "β = 300.0, σ1 = 365/14, ξ = 0.0"
run_dynamic_case "dynamic_Beta300Sigma1_14Days_BASELINE_NOVACCINE" "β = 300.0, σ1 = 365/14, qMax = 0.0"
run_dynamic_case "dynamic_Beta300Sigma1_14Days_VACCINE_LINEAR_K_0_025" "β = 300.0, σ1 = 365/14, vaccineCostProfile = :linear_k, ξKMin = 0.0, ξKMax = 0.25"








# run_dynamic_case "dynamic_T2_xi0001" "T_End = 2.0, ξ = 0.001"
# run_dynamic_case "dynamic_T3_xi0001" "T_End = 3.0, ξ = 0.001"

wait
echo "All Julia jobs completed."
notify_done "Le simulazioni su vulcano hanno finito!"

# ---------------------------------------------------------------------------
# Plot only from already-saved CSVs.
#
# These examples do not re-run the model. They reconstruct a result-like object
# from CSV files saved by run(...; outdir=...) or run_dynamic(...; outdir=...),
# then call save_all_figures.
#
# Important: pass a parameter object Pwith the same grid/calibration used to
# create the CSVs.

# julialauncher --project=. -e 'include("main.jl"); p = EpiEconMFG.MFGEpiEcon(); result = EpiEconMFG.load_solution_csv("outputs/dynamic_BASELINE", p); EpiEconMFG.save_all_figures(result, p; outdir="outputs/dynamic_BASELINE")'


# julialauncher --project=. -e 'include("main.jl"); p = EpiEconMFG.MFGEpiEcon(β = 400.0); result = EpiEconMFG.load_solution_csv("outputs/dynamic_Beta400", p); EpiEconMFG.save_all_figures(result, p; outdir="outputs/dynamic_Beta400")'
# julialauncher --project=. -e 'include("main.jl"); p = EpiEconMFG.MFGEpiEcon(β = 600.0); result = EpiEconMFG.load_solution_csv("outputs/dynamic_Beta600", p); EpiEconMFG.save_all_figures(result, p; outdir="outputs/dynamic_Beta600")'
# julialauncher --project=. -e 'include("main.jl"); p = EpiEconMFG.MFGEpiEcon(σ1 = 365/14, σ3 = 365/14); result = EpiEconMFG.load_solution_csv("outputs/dynamic_Sigma1_14DaysSigma3_14Days", p); EpiEconMFG.save_all_figures(result, p; outdir="outputs/dynamic_Sigma1_14DaysSigma3_14Days")'
# julialauncher --project=. -e 'include("main.jl"); p = EpiEconMFG.MFGEpiEcon(σ1 = 365/21, σ3 = 365/21); result = EpiEconMFG.load_solution_csv("outputs/dynamic_Sigma1_21DaysSigma3_21Days", p); EpiEconMFG.save_all_figures(result, p; outdir="outputs/dynamic_Sigma1_21DaysSigma3_21Days")'
# julialauncher --project=. -e 'include("main.jl"); p = EpiEconMFG.MFGEpiEcon(β = 400.0, σ1 = 365/14); result = EpiEconMFG.load_solution_csv("outputs/dynamic_Beta400Sigma1_14Days", p); EpiEconMFG.save_all_figures(result, p; outdir="outputs/dynamic_Beta400Sigma1_14Days")'
