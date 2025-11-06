import ROOT
from utils.rdf_handling import define_column

missing_cols = {
    #"nGoodAK4":"Sum(goodAK4Jets)",
    #"nGoodAK8":"Sum(goodAK8Jets)",
    "sum_AK4_pt":"Sum(goodAK4Jets_pt)",
    "sum_AK8_pt":"Sum(goodAK8Jets_pt)",
    "met_fatjet_dphi":"abs(ROOT::VecOps::DeltaPhi(Met_phi,goodAK8Jets_phi))",
    "leadAK8_pt":"goodAK8Jets_pt[ptSortedGoodAK8Jets[0]]",
    "leadAK8_eta":"goodAK8Jets_eta[ptSortedGoodAK8Jets[0]]",
    "leadAK8_phi":"goodAK8Jets_phi[ptSortedGoodAK8Jets[0]]",
    "leadAK8_msoftdrop":"goodAK8Jets_msoftdrop[ptSortedGoodAK8Jets[0]]",
    "vbsj_deta":"abs(vbsj1_eta -vbsj2_eta)",
    "vbsj_Mjj":"( ROOT::Math::PtEtaPhiMVector(vbsj1_pt,vbsj1_eta,vbsj1_phi,vbsj1_m) + ROOT::Math::PtEtaPhiMVector(vbsj2_pt,vbsj2_eta,vbsj2_phi,vbsj2_m) ).M()",
    "ptsortedGoodAK4":"Argsort(-goodAK4Jets_pt)",
    "leadAK4_phi":"goodAK4Jets_phi[ptSortedGoodAK4Jets[0]]",
    "leadAK4_MET_dphi":"abs(ROOT::VecOps::DeltaPhi(Met_phi,leadAK4_phi))",
    "leadAK8_MET_dphi":"abs(ROOT::VecOps::DeltaPhi(Met_phi,leadAK8_phi))",
    "HMET_dphi":"abs(ROOT::VecOps::DeltaPhi(Met_phi,Higgs_phi))",
    "V1MET_dphi":"abs(ROOT::VecOps::DeltaPhi(Met_phi,V1_phi))",
    "nCentralAK4":"Sum(abs(goodAK4Jets_eta)<=2.4)", #due to a bug in rdf script vbs and AK4 are practically reverted
    "vbsj_dphi":"abs(ROOT::VecOps::DeltaPhi(vbsj1_phi, vbsj2_phi))",
    "HV1_dR":"ROOT::VecOps::DeltaR(V1_eta, Higgs_eta, V1_phi, Higgs_phi)",
    "vbsj1_dRH":"ROOT::VecOps::DeltaR(vbsj1_eta, Higgs_eta, vbsj1_phi, Higgs_phi)",
    "vbsj2_dRH":"ROOT::VecOps::DeltaR(vbsj2_eta, Higgs_eta, vbsj2_phi, Higgs_phi)",
    "vbsj1_dRV1":"ROOT::VecOps::DeltaR(vbsj1_eta, Higgs_eta, vbsj1_phi, Higgs_phi)",
    "vbsj2_dRV1":"ROOT::VecOps::DeltaR(vbsj2_eta, Higgs_eta, vbsj2_phi, Higgs_phi)",
    "dphi_diff":"max(max(HMET_dphi,V1MET_dphi),ROOT::VecOps::DeltaPhi(Higgs_phi, V1_phi)) - min(min(HMET_dphi,V1MET_dphi),ROOT::VecOps::DeltaPhi(Higgs_phi, V1_phi))",
}

def fill_in_missing_cols(df):
    for (col, definition) in missing_cols.items():
        df = define_column(df,col,definition)
        #print('fill in missing var',col)
    return df
