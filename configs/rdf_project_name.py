# stores cut for ntuples, to be used for rdf.Filter(cut_def) 
# project ={ project_name: [cut_name1, cut_name2...]}
# cut_definition = { 'cut_name' : 'cut_definition' }

def get_project_filter(project_name):
    """Return the array of cut definitions for a given project name"""
    if project_name not in projects:
        raise ValueError(f"Project name '{project_name}' not found.")
    return [cut_definition[filter_name] for filter_name in projects[project_name]]

def get_project_dict(project_name):
    ''' return {(cut_name):(cut_def)}'''
    if project_name not in projects:
        raise ValueError(f"Project name '{project_name}' not found.")
    return {
        filter_name : cut_definition[filter_name]
        for filter_name in projects[project_name]
    }

projects = {
    #'all' : [],
    #'MET_pt' : ['has_enough_jets_met','MET_pt'],
    #'MET_pt_sig' : ['has_enough_jets_met','MET_pt','MET_significance'],
    #'MET_pt_sig_bveto' : ['has_enough_jets_met','MET_pt','MET_significance','ak4_bveto'],
    #'MET_pt_sig_bveto_HV' : ['has_enough_jets_met','MET_pt','MET_significance','ak4_bveto','loose_H','loose_V1',],
    #'full_MET_cutflow' : ['has_enough_jets_met','MET_pt','MET_significance','ak4_bveto','loose_H','loose_V1','has_vbs'],
    #'MET_trigger': ['MET_trigger'],
    #'MET_trigger_cutflow':  ['MET_trigger','has_enough_jets_met','MET_pt','MET_significance','ak4_bveto','loose_H','loose_V1','has_vbs'],
    #'MET_trigger_cutflow_3AK8':  ['MET_trigger','has_enough_jets_met','MET_pt','MET_significance','ak4_bveto','loose_H','loose_V1','has_vbs','3AK8','3goodAK8'],
    #'preselection'  :['MET_trigger','has_enough_jets_met','has_vbs'],
    'MET_trigger_cutflow_tight': ['MET_trigger','has_enough_jets_met','has_vbs','MET_pt','MET_significance','ak4_bveto','loose_H','loose_V1','tight_H','tight_V1'],
    'MET_trigger_cutflow_medium': ['MET_trigger','has_enough_jets_met','has_vbs','MET_pt','MET_significance','ak4_bveto','loose_H','loose_V1','medium_H','medium_V1'],
    'MET_trigger_cutflow_loose': ['MET_trigger','has_enough_jets_met','has_vbs','MET_pt','MET_significance','ak4_bveto','loose_H','loose_V1'],
    'vbscuts':['MET_trigger','has_enough_jets_met','has_vbs','MET_pt','MET_significance','ak4_bveto','tight_H','tight_V1','H_mass','V1_mass','vbsdeta_6','vbs_mjj'],
    'object_sel':['MET_trigger','has_enough_jets_met','has_vbs','MET_pt','ak4_bveto'],
    #'MET_trigger_cutflow_V1mass': ['MET_trigger','has_enough_jets_met','has_vbs','MET_pt','MET_significance','ak4_bveto','loose_H','loose_V1','V1_mass'],
    #'MET_trigger_cutflow_HV1mass': ['MET_trigger','has_enough_jets_met','has_vbs','MET_pt','MET_significance','ak4_bveto','loose_H','loose_V1','V1_mass','H_mass'],
    #'MET_trigger_cutflow_mt': ['MET_trigger','has_enough_jets_met','has_vbs','MET_pt','MET_significance','ak4_bveto','loose_H','loose_V1','between_med_tight_score'],
    #'MET_trigger_cutflow_loose_lowsig': ['MET_trigger','has_enough_jets_met','has_vbs','MET_pt_200','MET_significance_50','ak4_bveto','loose_H','loose_V1'],
    #'vbs_hadronic_enough_jets' : ['has_vbs_pair','has_enough_jets_allhad'],
    #'vbs_hadronic_enough_jets_hasH' : ['has_vbs_pair','has_enough_jets_allhad','loose_H'],
    #'vbs_hadronic_enough_jets_hasV1' : ['has_vbs_pair','has_enough_jets_allhad','loose_V1'],
    #'vbs_hadronic_enough_jets_hasV2' : ['has_vbs_pair','has_enough_jets_allhad','has_V2'],
    #'vbs_hadronic_enough_jets_hasHV1' : ['has_vbs_pair','has_enough_jets_allhad','loose_H','loose_V1'],  #missing this to full had for sig
    #'vbs_hadronic_enough_jets_hasV1V2' : ['has_vbs_pair','has_enough_jets_allhad','loose_V1','has_V2'],
    #'vbs_hadronic_enough_jets_hasVVH' : ['has_vbs_pair','has_enough_jets_allhad','loose_H','loose_V1','has_V2'],
    #'full_hadronic_cuts' : ['full_hadronic_cuts'],
    #'vbs' : ['has_vbs_pair'],
    #'vbs_PFHT' : ['has_vbs_pair','PFHT'],
    #'vbs_met_triga' : ['has_vbs_pair','met_trig_a'],
    #'vbs_met_trigb' : ['has_vbs_pair','met_trig_b'],
    #'vbs_met_enough_jets' : ['has_vbs_pair','has_enough_jets_met'],
    #'MET_enjet' : ['has_vbs_pair','has_enough_jets_met'],
    #'MET_pt_dphi' : ['has_vbs_pair','MET_pt','MET_dphi'],
    #'MET_pt_dphi_enjet' : ['has_vbs_pair','MET_pt','MET_dphi','has_enough_jets_met'],
    #'MET_hasH_pt_dphi' : ['has_vbs_pair','MET_pt','MET_dphi','has_enough_jets_met','loose_H'],
    #'MET_hasHV_pt_dphi' : ['has_vbs_pair','MET_pt','has_enough_jets_met','loose_H','loose_V1'],
    #'full_MET_triga' : ['has_vbs_pair','MET_pt','MET_dphi','has_enough_jets_met','loose_H','loose_V1','met_trig_a'],
    #'full_MET_trigb' : ['has_vbs_pair','MET_pt','MET_dphi','has_enough_jets_met','loose_H','loose_V1','met_trig_b'],
    'checking_met300':['MET_trigger','has_enough_jets_met','ak4_bveto','met300_sel','met300_cf'],
    'MET_objsel':['MET_trigger','MET_pt','met_sig_20','has_vbs','ak4_bveto','loose_H','loose_V1'],

    'MET_met300_7':['objsel',"MET_pt_300", 'vbsdeta_6', "HMET_dphi_p6", "V1MET_dphi_p6", "HScore_p6","V1Score_p6","vbsj_Mjj_1250"],

    'MET_medcut':['objsel','MET_pt_300','vbsdeta_2p5','HMET_dphi_p6','V1MET_dphi_p6',"HScore_p6","V1Score_p6"],
    'MET_medcut_ver2':['objsel','MET_pt_300','vbsdeta_2p5','HMET_dphi_p6','V1MET_dphi_p6',"medium_H","medium_V1"],
    'MET_medcut_ver3':['objsel','MET_pt_200','met_sig_20','has_vbs','ak4_bveto','loose_H','loose_V1','vbsdeta_2p5','HMET_dphi_p6','V1MET_dphi_p6',"HScore_p6","V1Score_p6"],
    'MET_medcut_ver4':['objsel','MET_pt','met_sig_20','has_vbs','ak4_bveto','loose_H','loose_V1','vbsdeta_2p5','HMET_dphi_p6','V1MET_dphi_p6',"HScore_p6","V1Score_p6"],
    'MET_medcut_looseH':['objsel','MET_pt_300','met_sig_20','has_vbs','ak4_bveto','loose_H','loose_V1','vbsdeta_2p5','HMET_dphi_p6','V1MET_dphi_p6',"V1Score_p6"],
    
    'MET_medcut_nottx':['objsel','MET_pt_300','vbsdeta_2p5','HMET_dphi_p6','V1MET_dphi_p6',"HScore_p6","V1Score_p6",'nottx'],

    'MET_tight':['objsel',"MET_pt_300", 'vbsdeta_6', "HMET_dphi_p6", "V1MET_dphi_p6", "HScore_p6","V1Score_p6","vbsj_Mjj_1250"],
    'MET_tight_noMETpt':['objsel', 'vbsdeta_6', "HMET_dphi_p6", "V1MET_dphi_p6", "HScore_p6","V1Score_p6","vbsj_Mjj_1250"],
    'MET_tight_novbsdeta':['objsel',"MET_pt_300", "HMET_dphi_p6", "V1MET_dphi_p6", "HScore_p6","V1Score_p6","vbsj_Mjj_1250"],
    'MET_tight_noHMETdphi':['objsel',"MET_pt_300", 'vbsdeta_6',"V1MET_dphi_p6", "HScore_p6","V1Score_p6","vbsj_Mjj_1250"],
    'MET_tight_noV1METdphi':['objsel',"MET_pt_300", 'vbsdeta_6',"HMET_dphi_p6", "HScore_p6","V1Score_p6","vbsj_Mjj_1250"],
    'MET_tight_noHScore':['objsel',"MET_pt_300", 'vbsdeta_6', "HMET_dphi_p6", "V1MET_dphi_p6", "V1Score_p6","vbsj_Mjj_1250"],
    'MET_tight_noV1Score':['objsel',"MET_pt_300", 'vbsdeta_6', "HMET_dphi_p6", "V1MET_dphi_p6", "HScore_p6","vbsj_Mjj_1250"],
    'MET_tight_novbsjMjj':['objsel',"MET_pt_300", 'vbsdeta_6', "HMET_dphi_p6", "V1MET_dphi_p6", "HScore_p6","V1Score_p6"],

    'MET_objsel':['objsel'],
    'MET_medtight':['objsel','MET_pt_300','vbsdeta_2p5','HMET_dphi_p6','V1MET_dphi_p6',"HScore_p6","V1Score_p6",'very_tight_H'],
    'MET_ABCDnoscores':['objsel','MET_pt_300','vbsdeta_2p5','HMET_dphi_p6','V1MET_dphi_p6',"HScore_p6","V1Score_p6"],

    'c2v1p5_tight':['objsel_noMETsig','MET_pt_300','HMET_dphi_p65','V1MET_dphi_p65',"HScore_p6","V1Score_p5",'Higgs_PNet_TvsQCD'],
    'c2v1p5_tightv2':['objsel_noMETsig','MET_pt_300','Higgs_PNet_TvsQCD'],
    'c2v1p5_med':['objsel_noMETsig','MET_pt_200','vbsj_Mjj_500','vbsj_deta_2',"V1Score_p6",'nAK4_l12','Higgs_PNet_TvsQCD','dphi_diff_2p7'],
}

cut_definition = {
    'preselection'  : 'Pass_MetTriggers && nGoodAK8>=2 && vbs_idx_max_Mjj[0]!=-99',
    'has_vbs_pair' : 'n_vbs_pair >0',
    'has_enough_jets_allhad': 'nGoodAK4 + 2 * nGoodAK8 >= 6',
    'has_enough_jets_met' : 'nGoodAK8 >= 2',
    #'PFHT' : 'HLT_PFHT',
    'met_trig_a' : 'met_trig_set_a',
    'met_trig_b' :'met_trig_set_b',
    'has_V2' : 'V2Score > 0',
    'AK8_2' : 'nGoodAK8 == 2',
    'AK8_3' : 'nGoodAK8 == 3',
    #'full_hadronic_cuts' :'pass_event_filter','HLT_PFHT','nGoodAK4 + 2 * nGoodAK8 >= 8','n_vbs_pair > 0','HiggsScore > 0','V1Score > 0','V2Score > 0',
    #'full_hadronic_cuts' :'pass_event_filter','nGoodAK4 + 2 * nGoodAK8 >= 6','n_vbs_pair > 0','HiggsScore > 0','V1Score > 0','V2Score > 0',
    'MET_pt' : 'Met_pt>100',
    'MET_dphi' : 'met_fatjet_dphi > 0.4',
    'MET_significance' : 'Met_significance > 100',
    'MET_pt_200':'Met_pt>200',
    'MET_pt_250':'Met_pt>250',
    'MET_pt_300':'Met_pt>300',
    'MET_pt_500':'Met_pt>500',
    'MET_significance_50':'Met_significance > 50',
    'ak4_bveto' : 'Pass_NoAK4BJet',
    #'has_vbs' : 'vbs_idx_max_Mjj[0]!=-99',
    'has_vbs' : 'vbs_idx_max_Mjj[0]!=-99',
    'MET_trigger' : 'Pass_MetTriggers',
    '3AK8':'nAK8>=3',
    '3goodAK8':'nGoodAK8>=3',
    'loose_H' : 'HiggsScore > 0.1',
    'loose_V1' : 'V1Score > 0.1',
    'medium_H' : 'HiggsScore > 0.5',
    'medium_V1' : 'V1Score > 0.5',
    'HScore_p6' : 'HiggsScore > 0.6',
    'V1Score_p6' : 'V1Score > 0.6',
    'V1Score_p5' : 'V1Score > 0.5',
    'tight_H':'HiggsScore >0.9',
    'tight_V1':'V1Score >0.9',
    'very_tight_H':'HiggsScore > 0.97',
    'loose_scores':'HiggsScore > 0.1 && V1Score > 0.1',
    'med_scores':'HiggsScore > 0.5 && V1Score > 0.5',
    'tight_scores':'HiggsScore > 0.9 && V1Score > 0.9',
    'between_med_tight_score':'HiggsScore > 0.5 && HiggsScore < 0.9 && V1Score > 0.5 && V1Score < 0.9',
    'between_loose_med_score':'HiggsScore > 0.1 && HiggsScore < 0.5 && V1Score > 0.1 && V1Score < 0.5',
    'V1_mass':'V1_msoftdrop<150',
    'H_mass':'Higgs_msoftdrop<150',
    'vbsdeta_6':'vbsj_deta>6',
    'vbsdeta_5':'vbsj_deta>5',
    'vbsdeta_2p5':'vbsj_deta>2.5',
    'vbs_mjj':'vbsj_Mjj>1000',
    'vbsj_Mjj_1250':'vbsj_Mjj>1250',
    'met300_sel':'HiggsScore>0.1 && V1Score>0.1 && Met_pt>300 && Met_significance >20',
    'met300_cf':'vbsj_deta>6 && HMET_dphi>0.6 && V1MET_dphi>0.6 && HiggsScore>0.6 && V1Score>0.6 ',
    'met_sig_20':'Met_significance > 20',
    'HMET_dphi_p6':'HMET_dphi>0.6',
    'V1MET_dphi_p6':'V1MET_dphi>0.6',
    'HMET_dphi_p65':'HMET_dphi>0.65',
    'V1MET_dphi_p65':'V1MET_dphi>0.65',
    'objsel':'Pass_MetTriggers && Met_pt>100 && Met_significance > 20 && vbs_idx_max_Mjj[0]!=-99 && Pass_NoAK4BJet && HiggsScore > 0.1 && V1Score > 0.1',
    'nottx':'sample_type!="ttx"',

    'objsel_noMETsig':'Pass_MetTriggers && Met_pt>100 && Pass_NoAK4BJet && HiggsScore > 0.1 && V1Score > 0.1 && vbs_idx_max_Mjj[0]!=-99 && nGoodAK8>=2',
    'vbsj_Mjj_500':'vbsj_Mjj>500',
    'vbsj_deta_2':'vbsj_deta>2',
    'leadAK8_met_dphi_1p5': 'leadAK8_MET_dphi>1.5',
    'nAK4_l12': 'nAK4<12',
    'HMET_dphi_p3': 'HMET_dphi> 0.3',
    'V1MET_dphi_p3': 'V1MET_dphi> 0.3',
    'Higgs_PNet_TvsQCD': 'Higgs_particleNet_TvsQCD<0.5',
    'dphi_diff_2p7':'dphi_diff<2.7'
}

unblinding = ['MET_trigger','has_enough_jets_met','has_vbs','MET_pt','MET_significance','ak4_bveto','loose_H','loose_V1','met300_sel','met300_cf'] #for these cuts it is ok to look at the data

object_selections = ['MET_trigger','has_enough_jets_met','has_vbs']
