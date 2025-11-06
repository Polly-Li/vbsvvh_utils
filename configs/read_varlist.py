#handle kinematics variables
import csv
from configs.paths import varlist_basic as default_varlist

def create_ranges(needed_var_cat=[], var_list = default_varlist):
    '''return range[var] = [n_bin, low, up, unit] '''
    ranges = {}

    with open(var_list, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            if row['sample_type'] == 'all':
                if not len(needed_var_cat) or (row['category'] in needed_var_cat):
                    variable = row['variable']
                    n_bin = int(row['n_bin'])
                    low = float(row['low'])
                    up = float(row['up'])
                    unit = row.get('unit', '') 
                    if variable not in ranges:
                        ranges[variable] = [n_bin, low, up, unit] 

    return ranges

def create_cut(needed_var_cat=[], var_list = default_varlist):
    '''return range[var] = [n_bin, low, up, unit] '''
    ranges = {}

    with open(var_list, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)

        for row in csv_reader:
            if row['sample_type'] == 'all':
                if not len(needed_var_cat) or (row['category'] in needed_var_cat):
                    variable = row['variable']
                    n_bin = int(row['n_bin'])
                    low = float(row['low'])
                    up = float(row['up'])
                    cut = float(row['cut']) if row['cut'] is not None else None
                    unit = row.get('unit', '') 
                    if variable not in ranges:
                        ranges[variable] = [low, up,cut] 

    return ranges