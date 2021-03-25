import os
import ubelt as ub
import pathlib
import pandas as pd
import argparse
from glob import glob
import ubelt as ub

import pdb

FILE_LOCATION = pathlib.Path(__file__).parent.absolute()
ROOT_DIR = "/home/local/KHQ/david.russell/data/viame_dvc/public/Aerial/US_ALASKA_MML_SEALION"
EXTENSTION = ".jpg"
PIPELINE_DIR = os.path.abspath(os.path.join(FILE_LOCATION, "../pipelines"))
OUTPUT_DIR = os.path.abspath(os.path.join(FILE_LOCATION, "../output"))
IMAGE_LIST = os.path.abspath(os.path.join(FILE_LOCATION, "../temp/image_list.txt"))
METHOD = "utility_add_segmentations_watershed"

os.chdir(PIPELINE_DIR)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", default=METHOD)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args

def fix_frame_number(input_filename):
    print(input_filename)
    os.system(f"dvc unprotect {input_filename}")
    data = pd.read_csv(input_filename, names=range(16))
    filenames = data.iloc[:, 1].copy()
    unique_filenames = filenames.unique()
    unique_filenames = unique_filenames.tolist()
    unique_filenames = pd.Index(unique_filenames)
    filename_indices = [unique_filenames.get_loc(filename) for filename in filenames.tolist()]
    print(data)
    data.iloc[:, 2] = filename_indices
    print(data)
    data.to_csv(input_filename, header=False, index=False)

def compute(root_dir=ROOT_DIR, method=METHOD, output_dir=OUTPUT_DIR, image_list=IMAGE_LIST, extension=EXTENSTION,
            debug=False ):
    folders = list(sorted(filter(os.path.isdir, glob(os.path.join(root_dir, "*")))))
    for folder in folders:
        annotation_file = os.path.join(folder, "sealions_" + os.path.basename(folder) + "*.viame.csv")
        annotation_file = glob(annotation_file)
        if len(annotation_file) != 1:
            breakpoint()
        annotation_file=annotation_file[0]
        if not os.path.isfile(annotation_file):
            continue
        fix_frame_number(annotation_file)
        image_names = pd.read_csv(annotation_file, squeeze=True, usecols=[1]).unique()
        # TODO update this
        image_files = list(map( lambda x : os.path.join(folder, x), image_names))

        with open(image_list, "w") as fout:
            fout.write("\n".join(image_files))

        cmd_string = (f"kwiver runner {method}.pipe " +
             f" --setting input:video_filename='{image_list}' --setting " +
             f"detector_writer:file_name='{os.path.join(output_dir, method + '_' + os.path.basename(folder))}.csv' "
             f"--setting detection_reader:file_name='{annotation_file}'")

        if debug:
            cmd_string = "gdb -ex 'run' --args " + cmd_string
        ub.cmd(cmd_string, verbose=3)

args = parse_args()

compute(method=args.method, debug=args.debug)
#compute(method="utility_add_segmentations_grabcut")
