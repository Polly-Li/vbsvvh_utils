import uproot
import numpy as np
import sys
import argparse
import scipy
import os
import ROOT
import pandas as pd

from lheweights_arrays import names_c2v, names_c2zc2z
names = names_c2v

VERBOSE = False

def getPandasDF(rdf_):
    numpy_arrays = rdf_.AsNumpy(columns=["RegionA", "RegionB", "RegionC", "RegionD", "weight"])
    pandas_df = pd.DataFrame(numpy_arrays)
    pandas_df = pandas_df.astype({
        "RegionA": bool,
        "RegionB": bool,
        "RegionC": bool,
        "RegionD": bool,
        "weight": float
    })
    return pandas_df

def selectYear(rdf_, year):
    if year=="Run2":
        pass
    elif year==2016:
        rdf = rdf.Filter("is2016")
    elif year==2017:
        rdf = rdf.Filter("is2017")
    elif year==2018:
        rdf = rdf.Filter("is2018")
    else:
        sys.exit(f"Did not recognize year: {year}")
    return rdf_


def applyLHEReweightingWeight(rdf_, idx_c2v):
    if (idx_c2v != -1):
        rdf_ = rdf_.Redefine("weight", f"weight * LHEReweightingWeight[{idx_c2v}]")
    else:
        print(f"Not applying LHEReweightingWeight for idx_c2v={idx_c2v}")

    return rdf_

def max_var(variations):
    deviations = [abs(1 - x) for x in variations]
    return variations[deviations.index(max(deviations))]

def get_variation(idx_c2v, year, input_dir, correction=None, tree="Events"):
    if(VERBOSE): print(f"--> get_variation(idx_c2v = {idx_c2v}, correction={correction})")
    sig_file = f"{input_dir}/sig.root"
    
    rdf = ROOT.RDataFrame(tree, sig_file)
    rdf = selectYear(rdf, year)

    # apply LHE reweighting weight for given coupling point
    rdf = applyLHEReweightingWeight(rdf, idx_c2v)

    # FIXME: is this necessary?
    rdf = rdf.Filter("abs(weight) < 100")

    # nominal
    a = rdf.Filter("RegionA").Sum("weight").GetValue()
    b = rdf.Filter("RegionB").Sum("weight").GetValue()
    c = rdf.Filter("RegionC").Sum("weight").GetValue()
    d = rdf.Filter("RegionD").Sum("weight").GetValue()

    if(VERBOSE): print(a, b, c, d)
    if correction is None:
        return ["{:.5f}".format(round(i, 5)) for i in [a, b, c, d]]

    variations = []

    # up variation
    if any(x in correction for x in ["jec", "jer", "met_unclustered", "jmr", "jms"]):
        sig_file_up = f"{input_dir}/sig_{correction}_up.root"
        rdf = ROOT.RDataFrame(tree, sig_file_up)
        rdf = rdf.Define("weight_up", f"weight")
    else:
        rdf = rdf.Define("weight_up", f"weight * {correction}[1] / {correction}[0]")

    a2 = rdf.Filter("RegionA").Sum("weight_up").GetValue()
    b2 = rdf.Filter("RegionB").Sum("weight_up").GetValue()
    c2 = rdf.Filter("RegionC").Sum("weight_up").GetValue()
    d2 = rdf.Filter("RegionD").Sum("weight_up").GetValue()
    if(VERBOSE): print(a2, b2, c2, d2)

    variations.append([(a - a2) / a, (b - b2) / b, (c - c2) / c, (d - d2) / d])

    # down variation
    if any(x in correction for x in ["jec", "jer", "met_unclustered", "jmr", "jms"]):
        sig_file_down = f"{input_dir}/sig_{correction}_down.root"
        rdf = ROOT.RDataFrame(tree, sig_file_down)
        rdf = rdf.Define("weight_down", f"weight")
    else:
        rdf = rdf.Define("weight_down", f"weight * {correction}[2] / {correction}[0]")

    a3 = rdf.Filter("RegionA").Sum("weight_down").GetValue()
    b3 = rdf.Filter("RegionB").Sum("weight_down").GetValue()
    c3 = rdf.Filter("RegionC").Sum("weight_down").GetValue()
    d3 = rdf.Filter("RegionD").Sum("weight_down").GetValue()
    if(VERBOSE): print(a3, b3, c3, d3)

    variations.append([(a - a3) / a, (b - b3) / b, (c - c3) / c, (d - d3) / d])

    return ["{:.5f}".format(round(1 + abs(max_var(x)), 5)) for x in zip(*variations)]


def get_stats(idx_c2v, year, input_dir, tree="Events"):
    if(VERBOSE): print("--> get_stats()")
    sig_file = f"{input_dir}/sig.root"
    
    rdf = ROOT.RDataFrame(tree, sig_file)
    rdf = selectYear(rdf, year)
    rdf = applyLHEReweightingWeight(rdf, idx_c2v)
    df = getPandasDF(rdf)

    df_a = df.query("RegionA")
    df_b = df.query("RegionB")
    df_c = df.query("RegionC")
    df_d = df.query("RegionD")

    a = 1 + np.sqrt((df_a.weight ** 2).sum()) / df_a.weight.sum()
    b = 1 + np.sqrt((df_b.weight ** 2).sum()) / df_b.weight.sum()
    c = 1 + np.sqrt((df_c.weight ** 2).sum()) / df_c.weight.sum()
    d = 1 + np.sqrt((df_d.weight ** 2).sum()) / df_d.weight.sum()

    return ["{:.4f}".format(round(i, 4)) for i in [a, b, c, d]]

def get_data(year, input_dir, tree="Events"):
    if(VERBOSE): print("--> get_data()")
    data_file = f"{input_dir}/data.root"
        
    rdf = ROOT.RDataFrame(tree, data_file)
    rdf = selectYear(rdf, year)

    b = rdf.Filter("RegionB").Sum("weight").GetValue()
    c = rdf.Filter("RegionC").Sum("weight").GetValue()
    d = rdf.Filter("RegionD").Sum("weight").GetValue()

    # use percent point function of gamma distribution 
    # returns the values at which the cumulative probability is equal to the specified quantile, giving the upper and lower bounds of the 99.73% confidence interval
    up1 = round(scipy.stats.gamma.ppf(1- (1-0.9973) / 2, b+1)) #self.obs[1] + 3*(self.obs[1]**0.5) #up1 will give the upper bound of the 99.73% confidence interval for the count in region B, based on the observed count b.
    up2 = round(scipy.stats.gamma.ppf(1- (1-0.9973) / 2, c+1)) #self.obs[2] + 3*(self.obs[2]**0.5)
    up3 = round(scipy.stats.gamma.ppf(1- (1-0.9973) / 2, d+1)) #self.obs[3] + 3*(self.obs[3]**0.5)
    dn1 = round(scipy.stats.gamma.ppf((1 - 0.9973) / 2, b)) #self.obs[1] - 3*(self.obs[1]**0.5)
    dn2 = round(scipy.stats.gamma.ppf((1 - 0.9973) / 2, c)) #self.obs[2] - 3*(self.obs[2]**0.5)
    dn3 = round(scipy.stats.gamma.ppf((1 - 0.9973) / 2, d)) #self.obs[3] - 3*(self.obs[3]**0.5)

    return [b, c, d, up1, up2, up3, dn1, dn2, dn3]

def xbb_reweight(year, channel_tag):
    if (year == "2016preVFP"):
        return "CMS_"+channel_tag+"_bTagFitXbb_13TeV_16preVFP          lnN   -                  -                  -                  -                  1.03010     1.03010     1.03010     1.03010"
    elif (year == "2016postVFP"):
        return "CMS_"+channel_tag+"_bTagFitXbb_13TeV_16postVFP         lnN   -                  -                  -                  -                  1.01200     1.01200     1.01200     1.01200"
    elif (year == "2017"):
        return "CMS_"+channel_tag+"_bTagFitXbb_13TeV_17                lnN   -                  -                  -                  -                  1.02560     1.02560     1.02560     1.02560"
    elif (year == "2018"):
        return "CMS_"+channel_tag+"_bTagFitXbb_13TeV_18                lnN   -                  -                  -                  -                  1.07080     1.07080     1.07080     1.07080"
    else:
        return f"""CMS_{channel_tag}_bTagFitXbb_13TeV_16preVFP          lnN   -                  -                  -                  -                  1.03010     1.03010     1.03010     1.03010
CMS_{channel_tag}_bTagFitXbb_13TeV_16postVFP         lnN   -                  -                  -                  -                  1.01200     1.01200     1.01200     1.01200
CMS_{channel_tag}_bTagFitXbb_13TeV_17                lnN   -                  -                  -                  -                  1.02560     1.02560     1.02560     1.02560
CMS_{channel_tag}_bTagFitXbb_13TeV_18                lnN   -                  -                  -                  -                  1.07080     1.07080     1.07080     1.07080
"""


def createDatacard(idx_c2v, datacard_name, year, input_dir, channel_tag):
    if(VERBOSE): print("--> createDatacard()")
    sys = {
        "sig": get_variation(idx_c2v, year, input_dir),
        "data": get_data(year, input_dir),
        "stat_unc": get_stats(idx_c2v, year, input_dir)
    }

    variations = [
        ["CMS_LHE_weights_pdf_vbsvvh", "lnN", get_variation(idx_c2v, year, input_dir, "LHEWeights_pdf")],
        ["CMS_PrefireWeight_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "L1PreFiringWeight")],
        ["CMS_vbsvvh_puWeight", "lnN", get_variation(idx_c2v, year, input_dir, "pileup_weight")],
        ["CMS_vbsvvh_puJetID", "lnN", get_variation(idx_c2v, year, input_dir, "pileupid_weight")],
        ["CMS_scale_j_Absolute_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_absolute")],
        ["CMS_scale_j_Absolute_2016postVFP_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_absolute_2016postVFP")],
        ["CMS_scale_j_Absolute_2016preVFP_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_absolute_2016preVFP")],
        ["CMS_scale_j_Absolute_2017_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_absolute_2017")],
        ["CMS_scale_j_Absolute_2018_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_absolute_2018")],
        ["CMS_scale_j_BBEC1_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_bbec1")],
        ["CMS_scale_j_BBEC1_2016postVFP_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_bbec1_2016postVFP")],
        ["CMS_scale_j_BBEC1_2016preVFP_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_bbec1_2016preVFP")],
        ["CMS_scale_j_BBEC1_2017_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_bbec1_2017")],
        ["CMS_scale_j_BBEC1_2018_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_bbec1_2018")],
        ["CMS_scale_j_EC2_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_ec2")],
        ["CMS_scale_j_EC2_2016postVFP_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_ec2_2016postVFP")],
        ["CMS_scale_j_EC2_2016preVFP_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_ec2_2016preVFP")],
        ["CMS_scale_j_EC2_2017_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_ec2_2017")],
        ["CMS_scale_j_EC2_2018_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_ec2_2018")],
        ["CMS_scale_j_FlavorQCD_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_flavorqcd")],
        ["CMS_scale_j_HF_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_hf")],
        ["CMS_scale_j_HF_2016postVFP_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_hf_2016postVFP")],
        ["CMS_scale_j_HF_2016preVFP_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_hf_2016preVFP")],
        ["CMS_scale_j_HF_2017_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_hf_2017")],
        ["CMS_scale_j_HF_2018_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_hf_2018")],
        ["CMS_scale_j_RelativeBal_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_relativebal")],
        ["CMS_scale_j_RelativeSample_2016postVFP_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_relativesample_2016postVFP")],
        ["CMS_scale_j_RelativeSample_2016preVFP_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_relativesample_2016preVFP")],
        ["CMS_scale_j_RelativeSample_2017_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_relativesample_2017")],
        ["CMS_scale_j_RelativeSample_2018_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jec_relativesample_2018")],
        ["CMS_res_j_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jer")],
        ["CMS_metUncl_13Tev", "lnN", get_variation(idx_c2v, year, input_dir, "met_unclustered")],
        # ["CMS_jms_pnetreg_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jms")],
        # ["CMS_jmr_pnetreg_13TeV", "lnN", get_variation(idx_c2v, year, input_dir, "jmr")],
        # ["btagWeightDeepJet_HF_13Tev", "lnN", get_variation(idx_c2v, year, input_dir, "btagging_scale_factors_HF")],
        # ["btagWeightDeepJet_LF_13Tev", "lnN", get_variation(idx_c2v, year, input_dir, "btagging_scale_factors_LF")],
        # [channel_tag+"_bTagWeightXbb_13TeV_16preVFP", "lnN", get_variation(idx_c2v, year, input_dir, "particlenet_h_weight_2016preVFP")],
        # [channel_tag+"_bTagWeightXbb_13TeV_16postVFP", "lnN", get_variation(idx_c2v, year, input_dir, "particlenet_h_weight_2016postVFP")],
        # [channel_tag+"_bTagWeightXbb_13TeV_17", "lnN", get_variation(idx_c2v, year, input_dir, "particlenet_h_weight_2017")],
        # [channel_tag+"_bTagWeightXbb_13TeV_18", "lnN", get_variation(idx_c2v, year, input_dir, "particlenet_h_weight_2018")],
        # [channel_tag+"_qTagWeightXWqq_13TeV_16preVFP", "lnN", get_variation(idx_c2v, year, input_dir, "particlenet_w_weight_2016preVFP")],
        # [channel_tag+"_qTagWeightXWqq_13TeV_16postVFP", "lnN", get_variation(idx_c2v, year, input_dir, "particlenet_w_weight_2016postVFP")],
        # [channel_tag+"_qTagWeightXWqq_13TeV_17", "lnN", get_variation(idx_c2v, year, input_dir, "particlenet_w_weight_2017")],
        # [channel_tag+"_qTagWeightXWqq_13TeV_18", "lnN", get_variation(idx_c2v, year, input_dir, "particlenet_w_weight_2018")],
        ["PSWeight_FSR", "lnN", get_variation(idx_c2v, year, input_dir, "PSWeight_FSR")],
        ["PSWeight_ISR", "lnN", get_variation(idx_c2v, year, input_dir, "PSWeight_ISR")],
        ["LHEScaleWeight_muF", "lnN", get_variation(idx_c2v, year, input_dir, "LHEScaleWeight_muF")],
        ["LHEScaleWeight_muR", "lnN", get_variation(idx_c2v, year, input_dir, "LHEScaleWeight_muR")],
        ["LHEWeights_pdf", "lnN", get_variation(idx_c2v, year, input_dir, "LHEWeights_pdf")]
    ]



    datacard = f"""imax 4 number of channels
jmax 1 number of backgrounds
kmax * number of nuisance parameters
shapes * * FAKE
--------------------------------------------------------------------------------------------------------------------------------
bin                                                     A                  B                  C                  D                  
observation                                             1                  {sys["data"][0]}                 {sys["data"][1]}                 {sys["data"][2]}               
--------------------------------------------------------------------------------------------------------------------------------
bin                                                     A                  B                  C                  D                  A           B           C           D           
process                                                 TotalBkg_AllHad3FJ    TotalBkg_AllHad3FJ    TotalBkg_AllHad3FJ    TotalBkg_AllHad3FJ    TotalSig    TotalSig    TotalSig    TotalSig    
process                                                 1                  1                  1                  1                  0           0           0           0           
rate                                                    1                  1                  1                  1                  {sys["sig"][0]}     {sys["sig"][1]}     {sys["sig"][2]}     {sys["sig"][3]}    
--------------------------------------------------------------------------------------------------------------------------------
CMS_{channel_tag}_control_abcd_syst                  lnN   1.2                -                  -                  -                  -           -           -           -           
lumi_13TeV_correlated                             lnN   -                  -                  -                  -                  1.016       1.016       1.016       1.016       
CMS_{channel_tag}_signal_RegionA                     lnN   -                  -                  -                  -                  {sys["stat_unc"][0]}      -           -           -
CMS_{channel_tag}_signal_RegionB                     lnN   -                  -                  -                  -                  -           {sys["stat_unc"][1]}      -           -           
CMS_{channel_tag}_signal_RegionC                     lnN   -                  -                  -                  -                  -           -           {sys["stat_unc"][2]}      -           
CMS_{channel_tag}_signal_RegionD                     lnN   -                  -                  -                  -                  -           -           -           {sys["stat_unc"][3]}     
"""
    
    for var in variations: 
        datacard +=f"""
{var[0]:<50}  {var[1]:<5}  {"-":<18} {"-":<18} {"-":<18} {"-":<18} {var[2][0]:<10} {var[2][1]:<10} {var[2][2]:<10} {var[2][3]:<10}
"""
#     datacard += f"""
# CMS_LHE_weights_pdf_vbsvvh                        lnN   -                  -                  -                  -                  {sys["LHEWeights_pdf"][0]}     {sys["LHEWeights_pdf"][1]}     {sys["LHEWeights_pdf"][2]}     {sys["LHEWeights_pdf"][3]}
# CMS_PSWeight_FSR_vbsvvh                           lnN   -                  -                  -                  -                  {sys["PSWeight_FSR"][0]}     {sys["PSWeight_FSR"][1]}     {sys["PSWeight_FSR"][2]}     {sys["PSWeight_FSR"][3]}
# CMS_PSWeight_ISR_vbsvvh                           lnN   -                  -                  -                  -                  {sys["PSWeight_ISR"][0]}     {sys["PSWeight_ISR"][1]}     {sys["PSWeight_ISR"][2]}     {sys["PSWeight_ISR"][3]}
# CMS_PrefireWeight_13TeV                           lnN   -                  -                  -                  -                  {sys["PrefireWeight_13TeV"][0]}     {sys["PrefireWeight_13TeV"][1]}     {sys["PrefireWeight_13TeV"][2]}     {sys["PrefireWeight_13TeV"][3]}
# CMS_vbsvvh_puWeight                               lnN   -                  -                  -                  -                  {sys["vbsvvh_puWeight"][0]}     {sys["vbsvvh_puWeight"][1]}     {sys["vbsvvh_puWeight"][2]}     {sys["vbsvvh_puWeight"][3]}
# CMS_LHE_weights_scale_muF_vbsvvh                  lnN   -                  -                  -                  -                  {sys["LHEScaleWeight_muF"][0]}     {sys["LHEScaleWeight_muF"][1]}     {sys["LHEScaleWeight_muF"][2]}     {sys["LHEScaleWeight_muF"][3]}
# CMS_LHE_weights_scale_muR_vbsvvh                  lnN   -                  -                  -                  -                  {sys["LHEScaleWeight_muR"][0]}     {sys["LHEScaleWeight_muR"][1]}     {sys["LHEScaleWeight_muR"][2]}     {sys["LHEScaleWeight_muR"][3]}
# CMS_vbsvvh_puJetID                                lnN   -                  -                  -                  -                  {sys["vbsvvh_puJetID"][0]}     {sys["vbsvvh_puJetID"][1]}     {sys["vbsvvh_puJetID"][2]}     {sys["vbsvvh_puJetID"][3]}
# """
    datacard += f"""
--------------------------------------------------------------------------------------------------------------------------------
A_{channel_tag} rateParam                  A  TotalBkg_{channel_tag}    (@0*@1/@2) B_{channel_tag},C_{channel_tag},D_{channel_tag}    
B_{channel_tag} rateParam                  B  TotalBkg_{channel_tag}    {sys["data"][0]} [{sys["data"][6]},{sys["data"][3]}]              
C_{channel_tag} rateParam                  C  TotalBkg_{channel_tag}    {sys["data"][1]} [{sys["data"][7]},{sys["data"][4]}]    
D_{channel_tag} rateParam                  D  TotalBkg_{channel_tag}    {sys["data"][2]} [{sys["data"][8]},{sys["data"][5]}]
"""

    output_dir = f"datacards/{os.path.basename(os.path.normpath(input_dir))}/{year}/"

    # check if directory for year exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(f"{output_dir}/{datacard_name}.dat" , "w") as f:
        f.write(datacard)

#---------------------------------------------------------------------------------
#---------------------------------------------------------------------------------
        
if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--indir", type=str, help="Directory with input files stored as sig.root and data.root")
    argparser.add_argument("--chtag", type=str, help="Channel name required to have unique names for rate parameters and certain uncertainties")
    argparser.add_argument("--output", type=str, default="-1", help="Output file name")
    argparser.add_argument("--year", type=str, default="Run2", help="Year of data taking")
    argparser.add_argument("--lhe_tag", type=str, default="c2v", help="Tag for LHE reweighting vector")
    args = argparser.parse_args()

    # Produce a datacard for each LHEReweighting point
    myrange=list(range(0, len(names)-1))
    if args.lhe_tag == "c2v":
        names = names_c2v
        myrange.insert(names.index("scan_CV_1p0_C2V_2p0_C3_1p0"), -1)
    elif args.lhe_tag == "c2wc2z":
        names = names_c2zc2z
        myrange.insert(names.index("scan_C2W_2p0_C2Z_2p0"), -1)
    
    for idx_c2v in myrange:
        datacard_name = names[myrange.index(idx_c2v)]
        if(VERBOSE): print(f"idx_c2v = {idx_c2v}, datacard_name={datacard_name}")
        createDatacard(idx_c2v, datacard_name, args.year, args.indir, args.chtag)
        
