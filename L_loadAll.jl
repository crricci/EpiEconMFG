using Pkg; Pkg.activate(@__DIR__)

using LinearAlgebra
using SparseArrays
using Parameters

using Plots

using JuMP
using Roots # for root finding in wage solver 
using DifferentialEquations

using OhMyREPL
using ProgressMeter
using BenchmarkTools
using LoopVectorization

include("L_parameters.jl")
include("L_diff.jl")
include("L_HJBsolver.jl")
include("L_wageSolver.jl")
include("L_aggregateVariables.jl")
include("L_KFEsolver.jl")
include("L_plots.jl")
include("L_MFGCsolver.jl")