
import sys, os

 # general purposed functions
def create_folder(file_name):
    if not os.path.exists(file_name):
        os.makedirs(file_name, exist_ok=True)

# csv and json 
def json_to_dict(file_name):
    '''load json file as dict'''
    import json
    with open(file_name, 'r') as f:
        data = json.load(f)
    return data

def csv_to_arr(file_name):
    """Return a 2D list (array) from a CSV file, converting numbers automatically."""
    import csv

    def try_convert(value):
        """Convert to int or float if possible; otherwise keep as string."""
        try:
            if '.' in value or 'e' in value.lower():
                return float(value)
            return int(value)
        except ValueError:
            return value

    data = []
    with open(file_name, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            if not any(row):
                continue
            converted_row = [try_convert(v.strip()) for v in row]
            data.append(converted_row)

    return data
    

def save_dict_to_json(data_dict, file_name):
    import json
    from funcs.common import create_folder
    """save a (nested) json as json with given file path+name"""

    create_folder(file_name)

    try:
        with open(file_name, 'w') as f:
            json.dump(data_dict, f, indent=4)
        print(f"Dictionary successfully saved to '{file_name}'")
    except Exception as e:
        print(f"Failed to save dictionary to '{file_name}': {e}")
        
def save_array_to_csv(arr, file_name):
    import csv
    """
    Save a 2D array to a CSV file.
    Assumes arr[0] is the header row, arr[1:] are data rows.
    """
    with open(file_name, mode='w', newline='') as f:
        writer = csv.writer(f)
        for row in arr:
            writer.writerow(row)

#print thing nicely

def sig_fig(val, n=4):
    '''Format a number to n significant figures in scientific notation'''
    if val == 0:
        return f"+0.{'0'*(n-1)}e+00"
    else:
        return f"{val:+.{n-1}e}"

def print_table(array, header=None):
    '''Print a 2D array as a table. Format numeric entries in scientific notation with 4 sig. fig.'''
    
    from prettytable import PrettyTable
    table = PrettyTable()
    if header is not None:
        table.field_names = header

    for row in array:
        formatted_row = []
        for val in row:
            if isinstance(val, (int, float)):
                formatted_row.append(sig_fig(val))
            else:
                formatted_row.append(str(val))
        table.add_row(formatted_row)

    print(table)

def print_dict_table(data_dict, header=None):
    '''print dict. provide a header if needed
    header must contain all columns including key of dict'''

    from prettytable import PrettyTable
    # Get lengths of all value lists
    value_lengths = [len(v) for v in data_dict.values()]
    if len(set(value_lengths)) != 1:
        print("Error: Not all dictionary values have the same length!")
        from pprint import pprint
        print("len:",value_lengths)
        pprint(data_dict)
        sys.exit(1)

    value_len = value_lengths[0]  # All same, so just take one

    # Generate default header if not provided
    if header is None:
        header = ["Key"] + [f"Col{i+1}" for i in range(value_len)]

    table = PrettyTable()
    if header is not None:
        table.field_names = header

    for key, values in data_dict.items():
        row = [key] + values
        formatted_row = []
        for val in row:
            if isinstance(val, (int, float)):
                formatted_row.append(sig_fig(val))
            else:
                formatted_row.append(str(val))
        table.add_row(formatted_row)

    print(table)


