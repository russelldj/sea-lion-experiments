import argparse
import pandas as pd
import ubelt as ub
import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file_1")
    parser.add_argument("file_2")
    args = parser.parse_args()
    return args

args = parse_args()
def compare_csvs(file_1, file_2, ncol=16):
    df1 = pd.read_csv(file_1, names=range(ncol))
    df2 = pd.read_csv(file_2, names=range(ncol))

    print(f"{file_1}:\n {df1}")
    print(f"{file_2}:\n {df2}")
    print(df1.compare(df2))

compare_csvs(args.file_1, args.file_2)
pdb.set_trace()