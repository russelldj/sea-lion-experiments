from bioharn.io.viame_csv import ViameCSV
import kwcoco
import glob
import pdb
import os

OUTPUT_FOLDER = "/home/local/KHQ/david.russell/data/viame_dvc/public/Aerial/US_ALASKA_MML_SEALION"

def convert(in_path, out_path):
    dset = kwcoco.CocoDataset()

    # TODO: ability to map image ids to agree with another coco file
    csv = ViameCSV(in_path)
    csv.extend_coco(dset=dset)
    dset.dump(out_path, newlines=True)

if __name__ == "__main__":
    files = glob.glob("/home/local/KHQ/david.russell/experiments/NOAA/sealion_pixel/output/utility_add_segmentations_watershed*.csv")
    for f in files:
        year = os.path.splitext(os.path.basename(f))[0][36:]

        convert(f, f.replace(".csv", ".kwcoco.json"))
    print(files)