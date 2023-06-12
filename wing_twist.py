#! /usr/bin/env python3

# Wing twist script, to make elliptical lift Wing
# Application of Lane et al, AIAA 2010-1227
#
# Copyright 2022, Pie Aeronefs s.a.

import numpy as np
import matplotlib.pyplot as plt
import openvsp as vsp

vsp.VSPCheckSetup()

parms = {
    "filename":"UG-2.vsp3",
    "output_name":"wing_fuse_vlm.txt",
    "CL":0.75, #target lift
    "base_aoa" : 0.,
    "test_aoa": 1., #degrees
    "debug" : False,
    "run_sims": True,
    # You can set this to look at a different geometry. The default is the first one.
    "geom_name":"Main_wing_base",
    "magnitude":1., #test perturbation
    "geom_set":"Set_1",
    "method":0,#0=VLM, 1=panel
    }

def ellipse(y,b,CLb):
    assert y <= b/2.
    semi_major = b/2.
    semi_minor = CLb*2./(np.pi*semi_major)
    Clc = semi_minor/semi_major*np.sqrt(semi_major**2-y**2)
    # Clc = semi_minor*np.sin(np.arccos(y/b))
    try :
        assert y**2/semi_major**2 + Clc**2/semi_minor**2 < 1.01
        assert y**2/semi_major**2 + Clc**2/semi_minor**2 > 0.99
    except AssertionError :
        print("y:",y)
        print("semi_major:",semi_major)
        print("Clc:",Clc)
        print("semi_minor:",semi_minor)
        print("ellipse:",y**2/semi_major**2 + Clc**2/semi_minor**2)
        raise ValueError("Solved Ellipse does not match definition!")
    return Clc

def sim(
    geom,static,
    aoa=0.,geom_set="Set_0",
    name=None,method=0,debug=False,oswald=False):
    #NOTE assuming symmetrical planform
    sym = 1
    geom_set = vsp.GetSetIndex(geom_set)
    vsp.DeleteAllResults()
    #NOTE return as np.array!
    vsp.SetAnalysisInputDefaults("VSPAEROComputeGeometry")
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry","AnalysisMethod",[0]) #Vortex Lattice = 0, panel = 1
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry","Symmetry",[sym])
    vsp.SetIntAnalysisInput("VSPAEROComputeGeometry","GeomSet",[geom_set])
    compgeom_results = vsp.ExecAnalysis("VSPAEROComputeGeometry")

    vsp.SetAnalysisInputDefaults("VSPAEROSweep")
    vsp.SetIntAnalysisInput("VSPAEROSweep","AlphaNpts",[1])
    try :
        assert vsp.GetIntAnalysisInput("VSPAEROSweep","AlphaNpts") == (1,)
    except AssertionError :
        print(vsp.GetIntAnalysisInput("VSPAEROSweep","AlphaNpts"))
        raise ValueError("AlphaNpts not equal to 1")
    vsp.SetIntAnalysisInput("VSPAEROSweep","AnalysisMethod",[method]) #Vortex Lattice = 0, panel = 1
    vsp.SetDoubleAnalysisInput("VSPAEROSweep","Sref",[static["vals"]["S_ref"]])
    vsp.SetDoubleAnalysisInput("VSPAEROSweep","bref",[static["vals"]["b_ref"]])
    vsp.SetDoubleAnalysisInput("VSPAEROSweep","cref",[static["vals"]["c_ref"]])
    vsp.SetDoubleAnalysisInput("VSPAEROSweep","AlphaStart",[aoa])
    vsp.SetIntAnalysisInput("VSPAEROSweep","Symmetry",[sym])
    vsp.SetIntAnalysisInput("VSPAEROSweep","GeomSet",[geom_set])
    vsp.SetIntAnalysisInput("VSPAEROSweep","NCPU",[8]) #set to 8 by default

    if debug :
        vsp.PrintAnalysisInputs("VSPAEROSweep")
    results = vsp.ExecAnalysis("VSPAEROSweep")
    if name == None :
        name = "unnamed"
    vsp.WriteResultsCSVFile(results,f"{name}_twist_optimizer.csv")
    loads = vsp.FindResultsID("VSPAERO_Load")
    if oswald :
        oswald_val = vsp.GetDoubleResults(vsp.FindResultsID("VSPAERO_Polar"),"E")
        return loads,oswald_val[0]
    else :
        return loads

def find_Cl_dist(results,get_y=False) :
    Yavg = np.array(vsp.GetDoubleResults(results,"Yavg"))
    #TODO results is giving YAvg for all wings, not just main wing.  must prune!
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
    Clc = chord * cl_norm
    if get_y :
        return Clc,Yavg
    else :
        return Clc

def adjust_wing_twist(geom,twists,static,span_stations=None):
    if span_stations is None:
        span_stations = len(twists)
    else :
        try:
            assert span_stations == len(twists)
        except AssertionError:
            print("span_stations: ", span_stations)
            print("twists: ",twists)
            raise ValueError("span_stations does not match len(twists)+1!")
    for i in range(span_stations) :
        vsp.SetParmVal(geom,"Twist",f"XSec_{i+1}",twists[i])
    vsp.Update()

def generate_A(
    geom,static,aoa,CL,b,span_stations,
    magnitude=1.,geom_set="Set_0",debug=False):
    N = span_stations
    # set washout to zero for baseline :
    adjust_wing_twist(
        geom,list(np.zeros(span_stations)),
        static,span_stations=span_stations)
    sim_results_baseline = sim(
        geom,static,
        geom_set=geom_set,name="baseline")
    Clcy,Yavg = find_Cl_dist(sim_results_baseline,get_y=True)
    Clc_y_target = set_target_CL(CL,b,Yavg) * static["vals"]["c_ref"]
    B = Clc_y_target - Clcy
    sim_results_aoa = sim(
        geom,static,
        geom_set=geom_set,aoa=aoa,name=f"aoa{aoa}")
    del_Cl_y_aoa = find_Cl_dist(sim_results_aoa) - Clcy
    if debug :
        print("aoa_CL_dist : ",find_Cl_dist(sim_results_aoa))
        print("Baseline CL dist : ", Clcy)
        print("resultant A column : ",del_Cl_y_aoa)
    M = len(Yavg)

    twist_sim_results = None
    for i in range(span_stations):
        twists = np.zeros(span_stations)
        twists[i] = magnitude
        print("twist : ",twists)
        print(f"Twist run {i} of {len(twists)}")
        adjust_wing_twist(geom,twists,static,span_stations=span_stations)
        sim_results_twist = sim(
            geom,static,
            geom_set=geom_set,name=f"twist_{i}")
        twist_sim =find_Cl_dist(sim_results_twist) - Clcy
        if twist_sim_results is None :
            twist_sim_results = np.array([twist_sim])
        else :
            twist_sim_results = np.append(twist_sim_results,[twist_sim],axis=0)
    A = np.append([del_Cl_y_aoa],twist_sim_results,axis=0)

    print("N : ", N)
    print("M : ", M)
    print("A dimensions :", A.shape)

    return np.transpose(A),Clcy,B # A should be m by n

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

def set_target_CL(CL,b,Yavg):
    Clc_y_target = np.array([ellipse(i,b,CL*b) for i in Yavg])
    M = len(Yavg)
    #NOTE kludge to account for both symmetry and non-symmetry conditions
    # counter-intuitively, non-symmetry includes left and right
    if any(i<0 for i in Yavg):
        assert int(M/2) == M/2
        CL_calc_rght = -np.trapz(Clc_y_target[int(M/2):],x=Yavg[int(M/2):])
        CL_calc_left = np.trapz(Clc_y_target[:int(M/2)],x=Yavg[:int(M/2)])
        CL_calc = (CL_calc_left+CL_calc_rght)/b
    else :
        CL_calc = np.trapz(Clc_y_target,x=Yavg)/(0.5*b)
    try:
        assert CL_calc > 0.9*CL
        assert CL_calc < 1.1*CL
    except AssertionError :
        print("CL_ellpise :", CL_calc)
        print("CL target : ",CL)
        print("Yavg : ",Yavg)
        # print(len(Yavg))
        print("CL(y) : ",Clc_y_target)
        # print(len(Clc_y_target))
        raise ValueError(
            "Elliptical lift distribution does not match target CL:")
    return Clc_y_target

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
    span_stations = vsp.GetNumXSec(vsp.GetXSecSurf(geom,0))-1
    b = static["vals"]["b_ref"]
    S = static["vals"]["S_ref"]
    test_aoa = parms["test_aoa"]
    CL = parms["CL"]
    mag = parms["magnitude"]
    output_name = parms["output_name"]
    geom_set = parms["geom_set"]
    debug = parms["debug"]

    N = span_stations
    if debug :
        print("span_stations :",span_stations)

    if parms["run_sims"] :
        A,Clcy,B = generate_A(
            geom,static,test_aoa,CL,b,span_stations,
            geom_set=geom_set,magnitude=mag)
        Clc_y_target = B + Clcy
        try :
            assert np.sum(np.matmul(np.linalg.pinv(A),A)-np.identity(N+1)) < 0.0001
        except AssertionError:
            print("pinv(A)A: ",np.matmul(np.linalg.pinv(A),A))
            raise ValueError("Pinv(A)A is not the identity matrix!")
        except ValueError:
            print("A: ",A)
            print("A Shape :", A.shape)
            print("pinv(A): ",np.linalg.pinv(A))
            print("pinv(A): ",np.linalg.pinv(A).shape)
            print("Rank A: ", np.linalg.matrix_rank(A))
            raise ValueError("Pinv(A) is not compatible with A")
        try:
            x = np.matmul(np.linalg.pinv(A),B)
        except ValueError:
            print("A: ",A)
            print("A Shape :", A.shape)
            print("pinv(A): ",np.linalg.pinv(A))
            print("pinv(A): ",np.linalg.pinv(A).shape)
            print("B : ",B)
            raise ValueError("pinv(A) is incompatible with B")
        print("Solution Vector : ", x)
        final_aoa = x[0]
        final_twist = x[1:]
        # b vector is Cl*c_target - Cl*c_current

        # evaluate results
        if debug :
            print(final_aoa)
            print(final_twist)
        adjust_wing_twist(
            geom,final_twist,static)
        vsp.WriteVSPFile('opt_'+parms["filename"])

        # final = sim(geom,static,aoa=0.,name="optimized")
        final,oswald = sim(
            geom,static,
            aoa=final_aoa,name="optimized",
            oswald=True)
        final_perf,Yavg = find_Cl_dist(final,get_y=True)

        fig = plt.figure(0)
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(Yavg, Clcy, label='Baseline')
        ax.plot(Yavg, Clc_y_target, label='Target')
        ax.plot(Yavg, final_perf, label='Result')
        ax.legend()
        plt.show()

        print("Baseline CL distribution: ",Clcy)
        print("Target CL distribution: ",Clc_y_target)
        if debug :
            print("A*x : ", np.matmul(A,x))
        print("b vector : ",B)
        if debug :
            print("A*x - B : ", np.matmul(A,x)-B)
        print("A Matrix : ",A)
        if debug :
            print("Rank A : ", np.linalg.matrix_rank(A))
        # print("Rank At : ", np.linalg.matrix_rank(np.transpose(A)))
        print("x vector : ", x)
        print("Optimized AoA : ", final_aoa)
        print("Optimized Twist Vector : ", final_twist)
        print("results : ",final_perf)
        print("final difference : ",Clc_y_target - final_perf)
        print("final oswald efficiency : ",oswald)
        with open(output_name,"w") as f :
            f.write("\nA Matrix : ")
            f.write(np.array2string(A))
            f.write("\nX vector : ")
            f.write(np.array2string(x))
            f.write("\nb vector : ")
            f.write(np.array2string(B))
            f.write("\nOptimized AoA : ")
            f.write(str(final_aoa))
            f.write("\nOptimized Twist Vector : ")
            f.write(str(final_twist))
            f.write("\nResults : ")
            f.write(np.array2string(final_perf))
            f.write("\nDifference : ")
            f.write(np.array2string(Clc_y_target - final_perf))
            f.write("\nOswald Efficiency Factor : ")
            f.write(str(oswald))
if __name__ == "__main__":
    main(parms)
