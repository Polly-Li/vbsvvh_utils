# to convert the yield output json to a csv table then to a pdf
# use csv_to_table for second part

from utils.common_utils import json_to_dict, save_array_to_csv, print_table
from tools.csv_to_table import csv_to_table
from combine_utils.run_combine import run_combine
import argparse,os


def main(json,output_name,blinded=False):
    data = json_to_dict(json)
    data = data["yields"]
    cuts = list(data.keys())
    print(f"Cuts: {type(cuts)}, {cuts}")
    samples = list(data[cuts[0]].keys())
    #samples.remove('metric')
    samples.remove('Background')
    samples.insert(1, "Background")
    samples.remove('metric')
    samples.insert(2, "metric")
    samples.remove('punzi')
    #samples.insert(3, "punzi")
    samples.remove('punzi_p1')
    samples.insert(3,"med_limit")
    if blinded:
        samples.remove('data')

    #samples.insert(4, "punzi_p1")
    print(f"samples: {type(samples)}, {samples}")
    header = ['Cut']+samples
    #count_table = [header]
    yield_table = [header]
    med_limit = {}
    for c in cuts:
        #counts = [c] + [data[c][sample][0] for sample in samples]
        #count_table.append(counts)
        med_limit[c] = run_combine(data[c]["Signal"][0],data[c]["Signal"][0]+data[c]['Background'][0],data[c]["Signal"][0]+data[c]['Background'][0])
        data[c]["med_limit"] = [med_limit[c]['m']]
        print(f"Cuts {c}: limit result {med_limit[c]}")
        yields = [c] + [data[c][sample][0] for sample in samples]
        yield_table.append(yields)
    #print_table(count_table)
    print_table(yield_table)
    #save_array_to_csv(count_table,f"{output_name}_count.csv")
    save_array_to_csv(yield_table,f"{output_name}_yield.csv")
    header2 = ['Cut','2.5%','16%','50%','84%','97.5%']
    limit_table = [header2]
    for c in cuts:
        limit_table.append([c]+[med_limit[c]['ll'],med_limit[c]['l'],med_limit[c]['m'],med_limit[c]['u'],med_limit[c]['uu']])
    save_array_to_csv(limit_table,f"{output_name}_limit.csv")
    csv_to_table([f"{output_name}_yield.csv",f"{output_name}_limit.csv"],f"{output_name}.pdf",3)

    



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='convert yield json file from coffea framework to csv file and table in pdf')
    parser.add_argument('jsonFiles', nargs='?', default='', help = 'Input json (generated from coffea make_plot)')
    parser.add_argument('--outname','-o', default=None, help = 'Specfy output name if needed')
    parser.add_argument('--output_dir','-d', default=None, help = 'Specfy output folder if needed')
    parser.add_argument('--blinded', default=False, help = 'blinded')
    args = parser.parse_args()
    jsonFiles  = args.jsonFiles
    output_name = args.outname
    output_dir = args.output_dir
    if output_name is None:
        base_name = os.path.basename(jsonFiles)             # e.g., 'folder/test_file.json' → 'test_file.json'
        output_name = os.path.splitext(base_name)[0]        # e.g., 'test_file.json' → 'test_file'
    if output_dir is not None:
        output_name = output_dir + output_name
    
    print(f"set output name as {output_name}")
    main(jsonFiles,output_name)