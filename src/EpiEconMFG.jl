module EpiEconMFG

using LinearAlgebra
using SparseArrays
using Parameters
using Roots

using CairoMakie
using ProgressMeter

export MFGEpiEcon,
    wage,
    returns,
    vaccine_monetary_cost,
    initial_distribution,
    create_test_distribution,
    load_initial_distribution_csv,
    solveModel,
    simulate_FP,
    value_iterationHJB,
    value_iterationHJB_given_wage,
    T_HJB,
    build_HJB_linear_system,
    compute_FP_policies,
    compute_FP_policies_dynamic,
    compute_time_dependent_policies,
    build_FP_generator,
    solve_fp_forward_dynamic,
    stack_distribution,
    unstack_distribution,
    renormalize_distribution!,
    project_nonnegative!,
    compute_∂V_dk,
    compute_∂V_dk!,
    DerivDkCache,
    optimal_labor,
    optimal_labor_ALL,
    aggregate_labor_supply,
    aggregate_kapital,
    compute_aggregates_path,
    compute_prices_path,
    fixed_point_wage,
    T_wage,
    solve_hjb_backward,
    solveModelDynamic,
    save_all_figures,
    save_solution_csv,
    load_solution_csv

include("core/parameters.jl")
include("core/diff.jl")
include("core/aggregates.jl")

include("solvers/wage_legacy.jl")
include("solvers/hjb_stationary.jl")
include("solvers/fp_kfe.jl")
include("solvers/hjb_time_dependent.jl")
include("solvers/coupled_quasistatic.jl")
include("solvers/coupled_forward_backward.jl")

include("visualization/plots.jl")

end # module EpiEconMFG
