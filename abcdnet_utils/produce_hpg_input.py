# procude root file that pass all cuts in project
import ROOT
from funcs.df_handling import apply_filters, save_df, read_root, get_count, get_weight
from funcs.df_handling import encode_sample_name, flatten
from funcs.variables import create_ranges
from names.sample_names import years,sample_categories
from names.project_name import get_project_filter
from names.paths import ntuple_file,varlist_ML,working_dir
from names.missing_var import fill_in_missing_cols
from funcs.common import get_argv
from funcs.file_handling import save_array_to_csv
import json

flatten_df = True #this to avoid infer error in ABCD training
project_name = 'MET_ABCDnoscores'
coupling = 'nominal'
coupling_dict = {
    'nominal':'weight',
    'c2v1p5':'weight*LHEReweightingWeight[72]'
}
set_weight = False
def main(project, save_all):
    sample_categories = ['sig','bkg']

    filters = get_project_filter(project)
    if not save_all:
        print("only save var in varlist_ML")
        vars = list(create_ranges([],varlist_ML).keys())+['weight',"LHEReweightingWeight"]+['sample_code']
        save_columns = ROOT.std.vector('string')()
        for col in vars:
            save_columns.push_back(col)
    else:
        print("save all column")
        save_columns = [] #save all branches

    yield_result = [['sample_category','sample_year','count','yield']]
    print("apply filter:",filters)
    for sample_cat in sample_categories:
        for year in years:
            df = read_root(ntuple_file[year,sample_cat])
            df = fill_in_missing_cols(df) 
            if sample_cat =='sig':
                for name,c in coupling_dict.items():
                    df = df.Define(name,c)
                if set_weight:  
                    df = df.Redefine("weight",coupling_dict[coupling])
            count = get_count(df)
            weight = get_weight(df)
            print(f"before filter: year:{year} cat:{sample_cat} count:{count} yield:{weight}")
            df = apply_filters(df,filters)
            count = get_count(df)
            weight = get_weight(df)
            df = encode_sample_name(df)
            if flatten_df: #this to avoid infer error
                print("save single entry column only")
                save_columns = flatten(df)
            yield_result.append([sample_cat,year,count,weight])
            print(f"after filter: year:{year} cat:{sample_cat} count:{count} yield:{weight}")
            save_df(df,f"{working_dir}/cutroots/{project}",f"cut_{year}_{sample_cat}",save_columns)
            
    result_json = f'{working_dir}/cutroots/{project}/yield.json'
    save_array_to_csv(yield_result,result_json)



if __name__ =="__main__":
    #project,save_all = get_argv("project name, save_all?",2,[str,bool])
    
    save_all = True
    main(project_name,save_all)
    #projects = {
        #'MET_tight':['objsel',"MET_pt_300", 'vbsdeta_6', "HMET_dphi_p6", "V1MET_dphi_p6", "HScore_p6","V1Score_p6","vbsj_Mjj_1250"],
        #'MET_tight_noMETpt':['objsel', 'vbsdeta_6', "HMET_dphi_p6", "V1MET_dphi_p6", "HScore_p6","V1Score_p6","vbsj_Mjj_1250"],
        #'MET_tight_novbsdeta':['objsel',"MET_pt_300", "HMET_dphi_p6", "V1MET_dphi_p6", "HScore_p6","V1Score_p6","vbsj_Mjj_1250"],
        #'MET_tight_noHMETdphi':['objsel',"MET_pt_300", 'vbsdeta_6',"V1MET_dphi_p6", "HScore_p6","V1Score_p6","vbsj_Mjj_1250"],
        #'MET_tight_noV1METdphi':['objsel',"MET_pt_300", 'vbsdeta_6',"HMET_dphi_p6", "HScore_p6","V1Score_p6","vbsj_Mjj_1250"],
        #'MET_tight_noHScore':['objsel',"MET_pt_300", 'vbsdeta_6', "HMET_dphi_p6", "V1MET_dphi_p6", "V1Score_p6","vbsj_Mjj_1250"],
        #'MET_tight_noV1Score':['objsel',"MET_pt_300", 'vbsdeta_6', "HMET_dphi_p6", "V1MET_dphi_p6", "HScore_p6","vbsj_Mjj_1250"],
        #'MET_tight_novbsjMjj':['objsel',"MET_pt_300", 'vbsdeta_6', "HMET_dphi_p6", "V1MET_dphi_p6", "HScore_p6","V1Score_p6"],
    #}
    #projects = projects.keys()
    #projects = ['MET_objsel',"MET_medcut","MET_medtight"]
    #projects = ['MET_medcut_nottx']
    #for project in projects:
    #    main(project, save_all)
    
    '''
    root_file = [ntuple_file[year,'bkg'] for year in years]
    df = read_root(root_file)
    df = fill_in_missing_cols(df)
    df = encode_sample_name(df)
    c = get_count(df)
    w = get_weight(df)
    print(f'c = {c} w = {w:.3f}')
    filters = get_project_filter(project_name)
    save_columns = flatten(df)
    print(save_columns)
    print('filter ',filters)
    df = apply_filters(df,filters)
    print(f'after filter: c = {get_count(df)}, w = {get_weight(df)}')
    save_df(df,f"/home/users/pyli/VVHjj/outputs/cutroots/MET_c2v1p5/{project_name}/allbkg.root",save_columns)
    for i in save_columns:
        print(i)

    root_file_sig = [ntuple_file[year,'sig'] for year in years]
    df_sig = read_root(root_file_sig)
    df_sig = fill_in_missing_cols(df_sig)
    df_sig = encode_sample_name(df_sig)
    df_sig = df_sig.Redefine("weight","LHEReweightingWeight[72] * weight")
    c = get_count(df_sig)
    w = get_weight(df_sig)
    print(f'c = {c} w = {w:.3f}')
    filters = get_project_filter(project_name)
    save_columns = flatten(df_sig)
    print('filter ',filters)
    df_sig = apply_filters(df_sig,filters)
    print(f'after filter: c = {get_count(df_sig)}, w = {get_weight(df_sig)}')
    save_df(df_sig,f"/home/users/pyli/VVHjj/outputs/cutroots/MET_c2v1p5/{project_name}/allsig.root",save_columns)
    for i in save_columns:
        print(i)
        '''
    