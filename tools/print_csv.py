import argparse
from utils.common_utils import csv_to_arr, print_table

def main(file):
    arr = csv_to_arr(file)
    print_table(arr[1:],arr[0])

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='print out a csv nicely')
    parser.add_argument("input", help = "input root file")

    args = parser.parse_args()
    main(args.input)