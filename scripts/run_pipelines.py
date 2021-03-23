import os
import ubelt as ub
import pathlib
import pandas as pd
import argparse
from glob import glob

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
    args = parser.parse_args()
    return args

def compute(root_dir=ROOT_DIR, method=METHOD, output_dir=OUTPUT_DIR, image_list=IMAGE_LIST, extension=EXTENSTION ):
    folders = list(sorted(filter(os.path.isdir, glob(os.path.join(root_dir, "*")))))
    for folder in folders[:1]:
        annotation_file = os.path.join(folder, "sealions_" + os.path.basename(folder) + "*.viame.csv")
        annotation_file = glob(annotation_file)
        if len(annotation_file) != 1:
            breakpoint()
        annotation_file=annotation_file[0]
        if not os.path.isfile(annotation_file):
            continue

        image_names = pd.read_csv(annotation_file, squeeze=True, usecols=[1]).unique()
        image_files = list(map( lambda x : os.path.join(folder, x), image_names))

        with open(image_list, "w") as fout:
            fout.write("\n".join(image_files[:1]))
        func = os.system

        func(f"gdb --args kwiver runner {method}.pipe " +
             f" --setting input:video_filename='{image_list}' --setting " +
             f"detector_writer:file_name='{os.path.join(output_dir, method + '_' + os.path.basename(folder))}.csv' "
             f"--setting detection_reader:file_name='{annotation_file}'")

args = parse_args()

compute(method=args.method)
#compute(method="utility_add_segmentations_grabcut")