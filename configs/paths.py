#current dir and output folder
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir)
output_dir = os.path.join(base_dir, "output")

#output folder of rdf
root_file_folder = '/data/userdata/pyli/projects/VVHjj/analysis/ntuples/'
ntuple_file = {
    (sample_year, sample_category): f"{root_file_folder}/{sample_year}_{sample_category}.root"
    for sample_year in ['2016preVFP','2016postVFP','2017','2018']
    for sample_category in ['sig','bkg','data']
}

sample_name_csv_file = os.path.join(base_dir,'configs','sample_names.csv')

#varlist for rdf
varlist_folder = f"{base_dir}/configs/rdf_varlists/"
varlist = f'{varlist_folder}/varlist.csv'
varlist_basic = f'{varlist_folder}/varlist_basic.csv'
varlist_test = f'{varlist_folder}/varlist_test.csv'
varlist_add = f'{varlist_folder}/varlist_add.csv'
varlist_ABCD = f'{varlist_folder}/varlist_ABCD.csv'
varlist_todraw = f'{varlist_folder}/varlist_todraw.csv'
varlist_ML = f'{varlist_folder}/varlist_ML.csv'
varlist_ML2 = f'{varlist_folder}/varlist_ML2.csv'
varlist_MET300cf = f'{varlist_folder}/varlist_MET300cf.csv'

#for signal study
raw_sig_path = '/ceph/cms/store/user/mmazza/SignalGeneration/v2_merged/'
sig_type_names = ['VBSWWH_OS','VBSWWH_SS','VBSZZH','VBSWZH']
sig_year_names = {'2016preVFP'  :'16',
                  '2016postVFP' :'16APV',
                  '2017'        :'17',
                  '2018'        :'18'}
raw_sig_root = {
    (sig_type_name, year_key): f"{raw_sig_path}{sig_type_name}_VBSCuts_TuneCP5_RunIISummer20UL{year_val}-106X_privateMC_NANOGEN_v2/merged.root"
    for sig_type_name in sig_type_names
    for year_key, year_val in sig_year_names.items()
}

#for ML: hpg outputs
double_training_result_path = os.path.join(base_dir,'configs','ABCDnet','train_result_paths_double.yaml')
ML_output_path = '/home/users/pyli/outputs/ABCD_trained/'