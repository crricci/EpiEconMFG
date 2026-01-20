
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