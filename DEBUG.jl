

function create_test_distribution(p)
    St = 0.7 * ones(p.Nk) 
    It = 0.1 * ones(p.Nk)
    Ct = 0.1 * ones(p.Nk)
    Rt = 0.1 * ones(p.Nk)
    Mass = sum(St + It + Ct + Rt) *  p.Δk
    St .= St ./ Mass
    It .= It ./ Mass
    Ct .= Ct ./ Mass
    Rt .= Rt ./ Mass
    return (ϕSt = St, ϕIt = It, ϕCt = Ct, ϕRt = Rt)
end


include("L_loadAll.jl")
p = MFGEpiEcon(ω=0.1, maxitHJBvalue=3000, ωw=0.2, maxitWage=200, tolWage=1e-3, verbose=true)
Ft = create_test_distribution(p)

V = (
    VS = 0.4*sqrt.(p.k),
    VI = 0.3*sqrt.(p.k),
    VC = 0.2*sqrt.(p.k),
    VR = 0.5*sqrt.(p.k)
)


# debug fixed point iteration for wage
# ∂VS_log = ∂k_log(V.VS, p.Nk, p.Δk, p.ϵDkUp)
# ∂VI_log = ∂k_log(V.VI, p.Nk, p.Δk, p.ϵDkUp)
# ∂VC_log = ∂k_log(V.VC, p.Nk, p.Δk, p.ϵDkUp)
# ∂VR_log = ∂k_log(V.VR, p.Nk, p.Δk, p.ϵDkUp)   
# w_star = fixed_point_wage(V,(∂kVS=∂VS_log, ∂kVI=∂VI_log, ∂kVC=∂VC_log, ∂kVR=∂VR_log),Ft,p; w0 = p.w_start)
# # debug T_wage function
# w = LinRange(0.01, 100.0, 1000)
# wages = [T_wage(wi,V,(∂kVS=∂VS_log, ∂kVI=∂VI_log, ∂kVC=∂VC_log, ∂kVR=∂VR_log),Ft,p) for wi in w]
# plot(w, wages, label="T_wage(w) vs w", xlabel="w", ylabel="T_wage(w)", title="Fixed Point Iteration for Wage")


# # debug HJB solver
# V1,w1 = T_HJB(V, Ft, p; w0=15.0)
# V2,w2 = T_HJB(V1, Ft, p; w0=w1)
# V3,w3 = T_HJB(V2, Ft, p; w0=w2)
# V4,w4 = T_HJB(V3, Ft, p; w0=w3)

# plot(p.k,V.VS, label="VS", xlabel="k", ylabel="Value", title="HJB Operator T_HJB")
# plot(p.k,V1.VS, label="VS_new", xlabel="k", ylabel="Value", title="HJB Operator T_HJB")
# plot(p.k,V2.VS, label="VS_new2", xlabel="k", ylabel="Value", title="HJB Operator T_HJB")
# plot(p.k,V3.VS, label="VS_new3", xlabel="k", ylabel="Value", title="HJB Operator T_HJB")
# plot(p.k,V4.VS, label="VS_new4", xlabel="k", ylabel="Value", title="HJB Operator T_HJB")

Vsol = nothing
try
    println("DEBUG: Nk=$(p.Nk), Δk=$(p.Δk), ω=$(p.ω), tol=$(p.tolHJBvalue), maxit=$(p.maxitHJBvalue)")
    println("DEBUG: w_start=$(p.w_start), ϵDkUp=$(p.ϵDkUp)")
    global Vsol = value_iterationHJB(V, Ft, p)
catch e
    println("value_iterationHJB failed: $e")
end
 