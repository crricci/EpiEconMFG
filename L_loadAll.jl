using Pkg; Pkg.activate(@__DIR__)

using LinearAlgebra
using Parameters

using Plots

using JuMP
using DifferentialEquations

using OhMyREPL
using ProgressMeter
using BenchmarkTools
using LoopVectorization

include("L_parameters.jl")
include("L_diff.jl")
include("L_HJBsolver.jl")
include("L_aggregateVariables.jl")
include("L_plots.jl")