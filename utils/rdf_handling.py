import configs.paths as paths
import ROOT

def read_root(file_path,tree_name="Events"):
    """need to import ROOT in main script"""
    df = ROOT.ROOT.RDataFrame(tree_name,file_path)
    print("Read root file from",file_path)
    return df

def get_count(df):
    return df.Count().GetValue()

def get_weight(df,weight="weight"):
    return df.Sum(weight).GetValue()

def check_column(df, column_name):
    """return True if column is in df"""
    return column_name in df.GetColumnNames()

def define_column(df_, column_name, definition):
    """redefine if exist, else define"""
    if not check_column(df_, column_name):
        df = df_.Define(column_name,definition)
    else:
        df = df_.Redefine(column_name,definition)
    return df

def apply_filters(df_, filters):
    df = df_
    '''input: array of filters. apply all filters'''
    if filters is not None and len(filters) != 0:
        for filter in filters:
            df = df.Filter(filter)
    return df

def calculate_weight(df, sample_category,weight_SM = False):
    '''create a new column "event_weight" in df'''
    if check_column(df,"weight"):
        return df
    else:
        if weight_SM:
            if sample_category == 'sig':
                df = df.Define("weight","weight_per_event[60]") #"LHE * weight_by_year"
            elif sample_category == 'bkg':
                df = df.Define("weight","gen_weight * weight_by_year") #weight_by_year =xsec * 1000 * lumi / sumws
            elif sample_category == 'data':
                df = df.Define("weight","weight_by_year")
            else:
                print('Failed to calculate weight')
                sys.exit(1)
        else:
            if sample_category == 'sig':
                df = df.Define("weight","weight_by_year")
            elif sample_category == 'bkg':
                df = df.Define("weight","gen_weight * weight_by_year")
            elif sample_category == 'data':
                df = df.Define("weight","weight_by_year")
            else:
                print('Failed to calculate weight')
                sys.exit(1)

    return df

def encode_sample_name(df, sample_name_setting=paths.sample_name_csv_file):
    import pandas as pd

    # Load mapping file
    mapping_df = pd.read_csv(sample_name_setting)

    # Build the lookup code
    cpp_code = """
    int lookup_sample_code(const std::string &name) {
    """
    for _, row in mapping_df.iterrows():
        cpp_code += f'  if (name == "{row["sample_name"]}") return {int(row["sample_code"])};\n'
    cpp_code += "  return -1;\n}\n"

    # Inject into ROOT
    ROOT.gInterpreter.Declare(cpp_code)

    # Define column
    df = df.Define("sample_code", "lookup_sample_code(name)") # changed from sample name to name
    return df

def decode_sample_code(df, sample_name_setting=paths.sample_name_csv_file):
    """return df with new column sample name, which stores sample name"""
    import pandas as pd
    # Load mapping file
    mapping_df = pd.read_csv(sample_name_setting)
 
    # Build C++ lookup function
    if not hasattr(ROOT, "_decode_sample_code_declared"):
        cpp_code = """
        std::string decode_sample_code(int code) {
        """
        for _, row in mapping_df.iterrows():
            cpp_code += f'  if (code == {int(row["sample_code"])}) return "{row["sample_name"]}";\n'
        cpp_code += '  return "UNKNOWN";\n}\n'

        ROOT.gInterpreter.Declare(cpp_code)
        ROOT._decode_sample_code_declared = True  # marker attribute

    # Define new column
    df = df.Define("sample_name", "decode_sample_code(sample_code)")
    return df
