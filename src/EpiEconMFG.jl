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
    create_test_distribution,
    solveModel,
    simulate_FP,
    value_iterationHJB,
    value_iterationHJB_given_wage,
    T_HJB,
    build_HJB_linear_system,
    compute_FP_policies,
    build_FP_generator,
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
    fixed_point_wage,
    T_wage,
    save_all_figures

include("core/parameters.jl")
include("core/diff.jl")
include("core/aggregates.jl")

include("solvers/wage_legacy.jl")
include("solvers/hjb_stationary.jl")
include("solvers/fp_kfe.jl")
include("solvers/coupled_quasistatic.jl")

include("visualization/plots.jl")

end # module EpiEconMFG
