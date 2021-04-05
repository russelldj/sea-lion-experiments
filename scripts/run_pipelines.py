import argparse
import os
import pathlib
import pdb
from glob import glob

import pandas as pd
import ubelt as ub

from convert_outputs import convert
from utils import get_all_files

FILE_LOCATION = pathlib.Path(__file__).parent.absolute()
ROOT_DIR = (
    "/home/local/KHQ/david.russell/data/viame_dvc/public/Aerial/US_ALASKA_MML_SEALION"
)
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
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--validate-lengths", action="store_true")
    parser.add_argument("--folder-index", type=int)
    args = parser.parse_args()
    return args


def count_lines(filename):
    line_count = 0
    for _ in open(filename):
        line_count += 1
    return line_count


def fix_frame_number(input_filename):
    print(input_filename)
    os.system(f"dvc unprotect {input_filename}")
    data = pd.read_csv(input_filename, names=range(16))
    filenames = data.iloc[:, 1].copy()
    unique_filenames = filenames.unique()
    unique_filenames = unique_filenames.tolist()
    unique_filenames = pd.Index(unique_filenames)
    filename_indices = [
        unique_filenames.get_loc(filename) for filename in filenames.tolist()
    ]
    print(data)
    data.iloc[:, 2] = filename_indices
    data.dropna(how="all", axis=1, inplace=True)
    print(data)
    data.to_csv(input_filename, header=False, index=False)
    cmd_str = f"dvc --cd {os.path.dirname(input_filename)} add {os.path.basename(input_filename)}"
    print(cmd_str)
    os.system(cmd_str)


def get_annotation_file(folder):
    annotation_file = os.path.join(
        folder, "sealions_" + os.path.basename(folder) + "*.viame.csv"
    )
    annotation_file = glob(annotation_file)
    annotation_file = list(
        filter(lambda x: "watershed" not in x and "grabcut" not in x, annotation_file)
    )
    if len(annotation_file) != 1:
        pdb.set_trace()
    return annotation_file[0]


def validate_lengths(annotation_folder, output_folder):
    for a, output_file in zip(
        get_all_files(annotation_folder, isdir=True),
        get_all_files(output_folder, sort_key=os.path.basename),
    ):
        annotation_file = get_annotation_file(a)
        annotation_count = count_lines(annotation_file)
        output_count = count_lines(output_file)
        print(
            annotation_count - output_count,
            annotation_count,
            output_count,
            output_file,
            annotation_file,
        )


def compute(
    root_dir=ROOT_DIR,
    method=METHOD,
    output_dir=OUTPUT_DIR,
    run=True,
    image_list=IMAGE_LIST,
    extension=EXTENSTION,
    debug=False,
    convert_only=False,
    check_output_counts=False,
    folder_index=None,
    fix_frame=False,
):
    import pdb

    folders = get_all_files(root_dir, isdir=True)
    if folder_index:
        start_index = folder_index
        end_index = folder_index + 1
    else:
        start_index = 0
        end_index = None

    for folder in folders[start_index:end_index]:
        annotation_file = get_annotation_file(folder)
        if not os.path.isfile(annotation_file):
            continue

        if fix_frame:
            fix_frame_number(annotation_file)

        image_names = pd.read_csv(annotation_file, squeeze=True, usecols=[1]).unique()
        # TODO update this
        image_files = list(
            map(
                lambda x: os.path.join(folder, x)
                if "images" in x
                else os.path.join(folder, "images", x),
                image_names,
            )
        )

        with open(image_list, "w") as fout:
            fout.write("\n".join(image_files))

        exit()
        output_file = (
            os.path.join(output_dir, method + "_" + os.path.basename(folder)) + ".csv"
        )
        kwcoco_file = annotation_file.replace(".viame.csv", "_watershed.kwcoco.json")
        if convert_only:
            ub.cmd(f"dvc unprotect {kwcoco_file}", verbose=3)
            convert(output_file, kwcoco_file)
            # ub.cmd(f"cp {output_file}.csv {updated_file}", verbose=3)
            add_cmd = f"dvc --cd {os.path.dirname(kwcoco_file)} add {os.path.basename(kwcoco_file)}"
            print(add_cmd)
            ub.cmd(add_cmd, verbose=3)
            continue

        if run:
            if debug:
                cmd_string = "gdb -ex 'run' --args " + cmd_string

            cmd_string = (
                f"kwiver runner {method}.pipe "
                + f" --setting input:video_filename='{image_list}' --setting "
                + f"detector_writer:file_name='{output_file}' "
                f"--setting detection_reader:file_name='{annotation_file}'"
            )
            ub.cmd(cmd_string, verbose=3)

        if check_output_counts:
            annotation_count = 0
            output_count = 0
            for line in open(annotation_file):
                annotation_count += 1
            for line in open(output_file):
                output_count += 1
            print(
                f"annotation lines {annotation_count}, outputlines {output_count}, diff {annotation_count - output_count}"
            )


args = parse_args()
if args.validate_lengths:
    validate_lengths(
        "/home/local/KHQ/david.russell/data/viame_dvc/public/Aerial/US_ALASKA_MML_SEALION",
        args.output_dir,
        # "/home/local/KHQ/david.russell/data/NOAA_sea_lion/watershed",
    )
    exit()
compute(
    method=args.method,
    debug=args.debug,
    check_output_counts=True,
    run=args.run,
    folder_index=args.folder_index,
)
# compute(method="utility_add_segmentations_grabcut")
