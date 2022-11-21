#! /usr/bin/env python3

# Wing twist script, to make elliptical lift Wing
# Application of Lane et al, AIAA 2010-1227
#
# Copyright 2022, Pie Aeronefs s.a.

import numpy as np
import openvsp as vsp

parms = {
    "filename":"a-wing.vsp3",
    # "b" : 8., #span, taken from VSP
    "CL":0.9, #target lift
    "N": 15, # n of basis functions NOTE N MUST BE GREATER THAN M!!!
    "aoa": 5., #degrees
    # "M": 5, # m of twist stations,taken from vsp
    "spanwise_resolution" : -1, # in 10^x to avoid float stations
    "debug" : True,
    "adjust_span_stations" : False,
    "run_sims": True,
    # you can set this to look at a different geometry
    #"geom_name":"Main_wing_base",
    "magnitude":1.
    }


def bias_y(b,M=5,) :
    # use circular distribution
    pts = np.linspace(0,1,M+1)#[1:]
    y = np.array([np.sin(i*np.pi/2.)*b/2. for i in pts])
    assert y[-1] == b/2.
    return y

def round_y(y,resolution=-1) :
    b = y[-1].copy() #to avoid any unpleasantness
    y = np.array([np.floor(i*10**-resolution)*10**resolution for i in y])
    y[-1] = b
    return y

def ellipse(y,b,CLb):
    # returns target Cl for a given station, span, and CL
    # y replaces traditional x
    # b is span, major axis, remember /2
    # CL*S is half area
    # y is position
    # A_ellipse = pi*a*b
    # b = A_ellipse/(pi*a)
    assert y <= b/2.
    semi_major = b/2.
    semi_minor = CLb*2./(np.pi*semi_major)
    Clc = semi_minor/semi_major*np.sqrt(semi_major**2-y**2)
    try :
        assert y**2/semi_major**2 + Clc**2/semi_minor**2 < 1.001
        assert y**2/semi_major**2 + Clc**2/semi_minor**2 > 0.999
    except AssertionError :
        print("y:",y)
        print("semi_major:",semi_major)
        print("Clc:",Clc)
        print("semi_minor:",semi_minor)
        print("ellipse:",y**2/semi_major**2 + Clc**2/semi_minor**2)
        raise ValueError("Solved Ellipse does not match definition!")
    return Clc

def rbf(r,r0):
    #note, arbitrary function, can be replaced by other schemes
    return np.sqrt(r**2+r0**2)

def interp(x1,x2,x):
    assert type(x1) in (tuple,list)
    assert type(x2) in (tuple,list)
    dydx = (x2[1]-x1[1])/(x2[0]-x1[0])
    y = dydx*(x-x1[0])+x1[1]
    return y

def test_interp():
    assert interp([1,1],[2,2],1.5) == 1.5

def sub_twist(y,b,eta):
    return -rbf(abs(y-eta),b/2.) #inverted to test washout

def master_twist(y,b,aoa,eta,weights=None,magnitude=1.):
    if weights is None :
        weights = np.ones(len(eta))
    else :
        assert len(eta)+1 ==len(weights)
    N = len(eta)
    # w(y) = del SUM(k_i*phi(norm(y-eta)))
    # w = magnitude * sum([weights[i]*rbf(abs(y-eta[i]),b) for i in range(N)])
    w = {"aoa":weights[0]*aoa}
    for station in y :
        w[station] = magnitude * sum(
            [weights[i+1]*sub_twist(station,b,eta[i]) for i in range(N)])
    return w

def sim(geom,static,aoa=0.,name=None):
    vsp.DeleteAllResults()
    #NOTE return as np.array!
    vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry","AnalysisMethod",[0]) #Vortex Lattice = 0, panel = 1
    compgeom_results = vsp.ExecAnalysis("VSPAEROComputeGeometry")

    vsp.SetAnalysisInputDefaults("VSPAEROSweep")
    vsp.SetIntAnalysisInput("VSPAEROSweep","AnalysisMethod",[0]) #Vortex Lattice = 0, panel = 1
    vsp.SetDoubleAnalysisInput("VSPAEROSweep","Sref",[static["vals"]["S_ref"]])
    vsp.SetDoubleAnalysisInput("VSPAEROSweep","bref",[static["vals"]["b_ref"]])
    vsp.SetDoubleAnalysisInput("VSPAEROSweep","cref",[static["vals"]["c_ref"]])
    vsp.SetDoubleAnalysisInput("VSPAEROSweep","AlphaStart",[aoa])
    vsp.SetDoubleAnalysisInput("VSPAEROSweep","AlphaNpts",[1])
    vsp.SetDoubleAnalysisInput("VSPAEROSweep","Symmetry",[1])

    results = vsp.ExecAnalysis("VSPAEROSweep")
    if name == None :
        name = "unnamed"
    vsp.WriteResultsCSVFile(results,f"{name}_twist_optimizer.csv")
    loads = vsp.FindResultsID("VSPAERO_Load")
    #NOTE LOAD file looks to give load at panel center, offset from
    # assert isinstance(results, np.ndarray)
    return loads

def find_Cl_at_y(results,y) :
    # if not isinstance(results, np.ndarray):
    Yavg = np.array(vsp.GetDoubleResults(results,"Yavg"))
    Yavg = Yavg[np.where(Yavg>0)]
    cl = np.array(vsp.GetDoubleResults(results,"cl"))
    cl_norm = np.array(vsp.GetDoubleResults(results,"cl*c/cref"))
    cref = vsp.GetDoubleResults(results,"FC_Cref_")[0]
    chord = np.array(vsp.GetDoubleResults(results,"Chord"))
    try :
        assert sum(cl*chord/cref-cl_norm) > -0.001
        assert sum(cl*chord/cref-cl_norm) < 0.001
    except AssertionError :
        print("cl_norm : ",cl_norm)
        print("sum : ",sum(cl_norm))
        print("cl normalized",cl*chord/cref)
        print("sum : ", sum(cl*chord/cref))
        print(cl*chord/cref-cl_norm)
        print(sum(cl*chord/cref-cl_norm))
        raise ValueError("cl_norm does not match normalized CL!")
    #perhaps check that cref is actually cref elsewhere?
    Clcy = []
    assert y[0] == 0
    # print(Yavg)
    for station in y :
        #NOTE for some reason Yavg includes negative y!
        # index1 = (Yavg < station 0)[0][-1]
        if station == 0. :
            index1 = 0
        else :
            index1 = np.searchsorted(Yavg,station)
        index2 = index1+1
        if index2 >= Yavg.shape[0] : #exception for last station
            index2 = len(Yavg)-1
            index1 = index2 - 1
        x1 = (Yavg[index1],cl_norm[index1])
        x2 = (Yavg[index2],cl_norm[index2])
        # print(station)
        # print(x1)
        # print(x2)
        # print(interp(x1,x2,station))
        Clcy.append(interp(x1,x2,station)*cref)
    Clcy = np.array(Clcy[1:])
    # print(cl_norm)
    # print(Clcy)
    return Clcy


def adjust_wing_twist(geom,twists,static,M=None):
    if type(twists) is not list :
        raise TypeError("Single station twist not yet implemented") #TODO fix
    if M is None:
        M = len(twists)
    else :
        try:
            assert M == len(twists)
        except AssertionError:
            print("M: ", M)
            print("twists: ",twists)
            raise ValueError("M does not match len(twists)+1!")
    for i in range(M) :
        # if i == 0 :
        #     vsp.SetParmVal(geom,"Twist",f"XSec_{i+1}",twists[i])
        # else:
        vsp.SetParmVal(geom,"Twist",f"XSec_{i+1}",twists[i])
    vsp.Update()
    check_static_parms(static)



def generate_A(geom,static,aoa,y,b,eta,magnitude=1.,M=None):
    N = len(eta)
    if M is None:
        M = len(y) - 1
    else :
        assert y[0] == 0
        try :
            assert M == len(y)- 1
        except AssertionError :
            raise ValueError(f"M is {M} while len(y) is {len(y)}")
    sim_results_baseline = sim(geom,static,name="baseline")
    Clcy = find_Cl_at_y(sim_results_baseline,y)
    sim_results_aoa = sim(geom,static,aoa=aoa,name=f"aoa{aoa}")
    del_Cl_y_aoa = find_Cl_at_y(sim_results_aoa,y) - Clcy
    # print(Clcy)
    # print(find_Cl_at_y(sim_results_aoa,y))
    # print(del_Cl_y_aoa)
    # quit()
    twist_sim_results = None
    for e in eta :
        twists = [magnitude*sub_twist(i,b,e) for i in range(M)]
        print("eta = ",e,", twist : ",twists)
        adjust_wing_twist(geom,twists,static,M=M)
        sim_results_twist = sim(geom,static,name=f"twist_eta{e}")
        twist_sim =find_Cl_at_y(sim_results_twist,y) - Clcy
        # print(Clcy)
        # print(find_Cl_at_y(sim_results_twist,y))
        # print(twist_sim)
        if twist_sim_results is None :
            twist_sim_results = np.array([twist_sim])
        else :
            twist_sim_results = np.append(twist_sim_results,[twist_sim],axis=0)
    # sub_A = np.array(
    #     [Clcy-twist_sim_results[i] for i in range(N)]
    #     )
    A = np.append([del_Cl_y_aoa],twist_sim_results,axis=0)
    try :
        assert A.shape == (N+1,M)
    except AssertionError :
        print("N : ", N)
        print("M : ", M)
        print("A dimensions :", A.shape)
        raise ValueError("A matrix is not N+1 by M matrix")
    return A,Clcy

def get_static_parms(static_parm_ids):
    static_parms = {
        "S_ref" : vsp.GetParmVal(static_parm_ids["S_ref"]),
        "c_ref" : vsp.GetParmVal(static_parm_ids["c_ref"]),
        "b_ref" : vsp.GetParmVal(static_parm_ids["b_ref"]),
        }
    return static_parms

def check_static_parms(static):
    new_static = get_static_parms(static["ids"])
    for i in static["vals"] :
        try :
            assert static["vals"][i] > new_static[i]*0.9999999
            assert static["vals"][i] < new_static[i]*1.0000001
        except AssertionError :
            print("original ",i,static["vals"][i])
            print("new ",i,new_static[i])
            raise ValueError("Static values have changed!")

def main(parms):
    # initialize
    vsp.ClearVSPModel()
    vsp.ReadVSPFile(parms["filename"])
    if "geom_id" not in parms :
        if "geom_name" not in parms :
            geom = vsp.FindGeoms()[-1]
        else :
            geom = vsp.FindGeomsWithName(parms["geom_name"])[-1]
    else :
        geom = parms["geom_id"]

    # get geom parameters which must not change
    static_parm_ids = {
        "S_ref" : vsp.FindParm(geom,"TotalArea","WingGeom"),
        "c_ref" : vsp.FindParm(geom,"TotalChord","WingGeom"),
        "b_ref" : vsp.FindParm(geom,"TotalSpan","WingGeom"),
        }
    static = {"ids":static_parm_ids,"vals":get_static_parms(static_parm_ids)}

    # get parameters for script
    M = vsp.GetNumXSec(vsp.GetXSecSurf(geom,0))-1
    b = static["vals"]["b_ref"]
    S = static["vals"]["S_ref"]
    aoa = parms["aoa"]
    CL = parms["CL"]
    N = parms["N"]
    mag = parms["magnitude"]

    debug = parms["debug"]
    if debug :
        print("M :",M)
        test_interp()
        # print("test twist:",str([twist(i,4.) for i in range(5)]))
    # generate list of y spacings
    y = bias_y(b,M=M)
    y = round_y(y)
    spans = [y[i+1]-y[i] for i in range(M)]
    spans = [y[0]]+spans
    print("Spanwise Stations : ", y)
    # print(spans)
    try :
        assert sum(spans) > b*.5*0.9999999
        assert sum(spans) < b*.5*1.0000001
    except AssertionError:
        print("SubSpans: ",spans)
        print("Sum of spans: ",sum(spans))
        print("Semi-Span: ", b/2.)
        raise ValueError("Sum of spans not equal to semi-span!")
    if parms["adjust_span_stations"] :
        for i,span in enumerate(spans) :
            vsp.SetParmVal(geom,"Span",f"XSec_{i+1}",span)
        vsp.Update()
        check_static_parms(static)
        if not debug :
            vsp.WriteVSPFile(parms["filename"],SET_ALL)


    Clc_y_target = np.array([ellipse(i,b,CL*b) for i in y])
    if debug :
        print("Clc traget distribution :",Clc_y_target)

    # assert sum Clc_y_target == CL
    # NOTE : CL = integral Cl(b) *c/cref/bref, in this case, c/cref = 1.
    temp = sum([
        (y[i+1]-y[i])
        * (Clc_y_target[i+1]+Clc_y_target[i])*.5
        for i in range(M-1)])
    # CL_calc = (temp+y[0]*Clc_y_target[0])*2/b
    CL_calc = temp*2/b
    try:
        assert CL_calc > 0.9*CL
        assert CL_calc < 1.1*CL
    except AssertionError :
        print("CL_ellpise :", CL_calc)
        print("CL target : ",CL)
        raise ValueError(
            "Elliptical lift distribution does not match target CL:")

    if parms["run_sims"] :
        # get Cl*c(y) desired as vector

        eta = np.linspace(0,b/2.,N)
        A,Clcy = generate_A(geom,static,aoa,y,b,eta,magnitude=mag,M=M)
        # TODO solve for x
        B = Clc_y_target[1:] - Clcy
        assert B.shape[0] == M
        assert np.sum(np.matmul(np.linalg.pinv(A),A)-np.identity(M)) < 0.0001
        x = np.matmul(np.transpose(np.linalg.pinv(A)),B)
        print("Twist Weight Vector : ", x)
        final_twist = master_twist(y,b,aoa,eta,weights=x)
        # b vector is Cl*c_target - Cl*c_current

        # evaluate results
        if debug :
            print([final_twist[s] for s in final_twist][1:])
        adjust_wing_twist(geom,[final_twist[s] for s in final_twist][1:],static)
        # final = sim(geom,static,aoa=0.,name="optimized")
        final = sim(geom,static,aoa=final_twist["aoa"],name="optimized")
        final_perf = find_Cl_at_y(final,y)

        print("Baseline CL distribution: ",Clcy)
        print("Target CL distribution: ",Clc_y_target)
        print("b vector : ",B)
        print("A Matrix : ",A)
        print("Optimized Twist Vector : ", final_twist)
        print("results : ",final_perf)
        print("final difference : ",Clc_y_target[1:] - final_perf)
if __name__ == "__main__":
    main(parms)

