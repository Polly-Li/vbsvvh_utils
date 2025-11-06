# procude root file that pass all cuts in project
import ROOT
from utils.rdf_handling import apply_filters, save_df, read_root, get_count, get_weight
from utils.rdf_handling import encode_sample_name, flatten
from configs.read_varlist import create_ranges
from configs.rdf_sample_names import years,sample_categories
from configs.rdf_project_name import get_project_filter
from configs.paths import ntuple_file,varlist_ML,output_dir 
from configs.rdf_missing_var import fill_in_missing_cols
from utils.common_utils import save_array_to_csv, create_folder
import json

flatten_df = True #this to avoid infer error in ABCD training
project_name = 'MET_ABCDnoscores'
coupling = 'nominal'
coupling_dict = {
    'nominal':'weight',
    'c2v1p5':'weight*LHEReweightingWeight[72]'
}
set_weight = False # to redefine weight

def main(project, save_all):
    sample_categories = ['sig','bkg']
    output_folder = f'{output_dir}/cutroots/{project}'
    create_folder(output_folder)

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
            save_df(df,output_folder,f"cut_{year}_{sample_cat}",save_columns)
            
    result_json = f'{output_folder}/yield.json'
    save_array_to_csv(yield_result,result_json)



if __name__ =="__main__":
    
    save_all = True
    main(project_name,save_all)