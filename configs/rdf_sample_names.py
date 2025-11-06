# to store definitions of samples

#years
sample_years = ['2016preVFP','2016postVFP','2017','2018']
years = ['2016preVFP','2016postVFP','2017','2018']

#sample_categories
sample_categories = ['sig','bkg','data']

#sample_types
sample_types = {
    'sig' : ['WWH_OS', 'WWH_SS', 'WZH', 'ZZH'],
    #'bkg' : ['QCD', 'ttbar_had','ttbar_SL', 'ST', 'WJet', 'ZJet', 'EWKV', 'ttX', 'bosons'],
    'bkg' : ["ttbar","ttx","ST", "WJets","ZJets", "EWK","QCD","Other"],#no DY
    #'bkg' : ['QCD', 'ttbar', 'ST', 'WJet', 'ZJet', 'EWKV', 'ttX', 'bosons'],
    'data' : ['data']
}

#sample_names
sample_names = {
    'WWH_OS' : ["VBSWWH_OS_VBSCuts"],
    'WWH_SS' : ["VBSWWH_SS_VBSCuts"],
    'WZH' :  ["VBSWZH_VBSCuts"],
    'ZZH' : ["VBSZZH_VBSCuts"],
    "DY" : ["DY"],
    # "Other" :["Other"],
    # "ttx":["ttx"],
    # "WJets":["WJets"],
    # "ZJets":["ZJets"],
    # "EWK":["EWK"],
    'QCD' : ["QCD_HT50to100", "QCD_HT100to200", "QCD_HT200to300", "QCD_HT300to500",  "QCD_HT500to700" ,  "QCD_HT700to1000", "QCD_HT1000to1500", "QCD_HT1500to2000", "QCD_HT2000toInf","QCD"],
    'ttbar_had' : ["TTToHadronic"],
    'ttbar_SL':  ["TTToSemiLeptonic"],
    'ttbar' : ["TTToHadronic","TTToSemiLeptonic","ttbar"],
    'ST' : ["ST_t-channel_antitop_4f_InclusiveDecays" , "ST_t-channel_top_4f_InclusiveDecays" , "ST_tW_antitop_5f_inclusiveDecays" , "ST_tW_top_5f_inclusiveDecays", "ST"],
    'WJets' : ["WJetsToQQ_HT-200to400" , "WJetsToQQ_HT-400to600" , "WJetsToQQ_HT-600to800" , "WJetsToQQ_HT-800toInf", "WJets"],
    'ZJets' : ["ZJetsToQQ_HT-200to400" , "ZJetsToQQ_HT-400to600" , "ZJetsToQQ_HT-600to800" , "ZJetsToQQ_HT-800toInf","ZJets"],
    'EWK' : ["EWKWminus2Jets_WToQQ_dipoleRecoilOn" , "EWKWplus2Jets_WToQQ_dipoleRecoilOn" , "EWKZ2Jets_ZToLL_M-50" , "EWKZ2Jets_ZToNuNu_M-50" , "EWKZ2Jets_ZToQQ_dipoleRecoilOn"], #was called EWKV
    'ttx' : ["TTWW" , "TTWZ" , "TTWJetsToQQ", "ttHToNonbb_M125" , "ttHTobb_M125" , "TTbb_4f_TTToHadronic"],  #was called ttX
    'Other' :["VHToNonbb_M125" , "WWTo1L1Nu2Q_4f" , "WWTo4Q_4f" , "WWW_4F" , "WWZ_4F" , "WZJJ_EWK_InclusivePolarization" , "WZTo1L1Nu2Q_4f" , "WZTo2Q2L_mllmin4p0" , "WZZ" , "WminusH_HToBB_WToLNu_M-125" , "WplusH_HToBB_WToLNu_M-125" , "ZH_HToBB_ZToQQ_M-125" , "ZZTo2Nu2Q_5f" , "ZZTo2Q2L_mllmin4p0" , "ZZTo4Q_5f" , "ZZZ"], #was called bosons
    "data" : ['MET'],
}
def get_sample_colour(sample_name):
    import ROOT
    sample_colour = {
        #data
        "data"  : ROOT.kBlue,

        #sig 
        "sig"   : ROOT.kRed,
        'WWH_OS' : ROOT.kRed+1,
        'WWH_SS' : ROOT.kRed+2,
        'WZH' : ROOT.kRed+3,
        'ZZH' : ROOT.kRed+4,

        #bkg
        "bkg_total": ROOT.kGray+3,
        "DY" : ROOT.kGreen-7,
        'ttbar' : ROOT.kTeal+7,
        'ttbar_had' : ROOT.kCyan-8,
        'ttbar_SL':  ROOT.kCyan+3,
        "ttx" : ROOT.kTeal+3,
        'ttX' : ROOT.kTeal+3,
        'ST' : ROOT.kAzure-5,
        "WJets" : ROOT.kOrange+6,
        'WJet' : ROOT.kOrange+6,
        "ZJets" :ROOT.kYellow-9,
        'ZJet' : ROOT.kYellow-9,
        'EWKV' : ROOT.kAzure+1,
        "EWK": ROOT.kAzure+1,
        'QCD' : ROOT.kCyan-10,
        'bosons' :ROOT.kOrange+4,
        "Other" : ROOT.kMagenta-10,
    }
    return sample_colour[sample_name]

def oring_sample_names(sample_type):
    """Return a string formatted as 'name1 || name2 || ...' for names listed in array sample_name."""
    if sample_type not in sample_names:
        raise ValueError(f"Sample name '{sample_type}' not found in sample_names dictionary.")
    
    return " || ".join(f"sample_name == \"{name}\"" for name in sample_names[sample_type])

def filter_sample_type(df,sample_type):
    '''return df.Filter(sample_type) e.g. QCD'''
    return df.Filter(oring_sample_names(sample_type))

