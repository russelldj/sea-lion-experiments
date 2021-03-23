import os
import ubelt as ub
import pathlib
from glob import glob

FILE_LOCATION = pathlib.Path(__file__).parent.absolute()
ROOT_DIR = "/home/local/KHQ/david.russell/data/viame_dvc/public/Aerial/US_ALASKA_MML_SEALION"
EXTENSTION = ".jpg"
PIPELINE_DIR = os.path.abspath(os.path.join(FILE_LOCATION, "../pipelines"))
OUTPUT_DIR = os.path.abspath(os.path.join(FILE_LOCATION, "../output"))
IMAGE_LIST = os.path.abspath(os.path.join(FILE_LOCATION, "../temp/image_list.txt"))
METHOD = "utility_add_segmentations_watershed"

os.chdir(PIPELINE_DIR)

def compute(root_dir=ROOT_DIR, method=METHOD, output_dir=OUTPUT_DIR, image_list=IMAGE_LIST, extension=EXTENSTION ):
    files = list(sorted(filter(os.path.isdir, glob(os.path.join(root_dir, "*")))))
    print(files)
    for f in files:
        image_files = glob( os.path.join(f, "images", "*.jpg") )
        annotation_file = os.path.join(f, "sealions_" + os.path.basename(f) + "_v9.viame.csv")
        if not os.path.isfile(annotation_file):
            continue
        print(annotation_file)
        with open(image_list, "w") as fout:
            fout.write("\n".join(image_files[:3]))
        func = print

        func(f"kwiver runner {method}.pipe " +
             f" --setting input:video_filename='{image_list}' --setting " +
             f"detector_writer:file_name='{os.path.join(output_dir, method + '_' + os.path.basename(f))}.csv' "
             f"--setting detection_reader:file_name='{annotation_file}'")

compute(method="utility_add_segmentations_watershed")
compute(method="utility_add_segmentations_grabcut")