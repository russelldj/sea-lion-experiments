#!/bin/sh
# Input locations and types
export INPUT_DIRECTORY=training_data

# Path to VIAME installation
export VIAME_INSTALL="/home/local/KHQ/david.russell/dev/VIAME/build_debug/install"

source ${VIAME_INSTALL}/setup_viame.sh

# Run pipeline
viame_train_detector \
  -i ${INPUT_DIRECTORY} \
  -c train_adaboost_pixel_classifier.viame_csv.conf \
  --threshold 0.0
#
#gdb -ex "set breakpoint pending on" \
#	-ex run --args kwiver runner filter_burnout.pipe -s input:video_filename=input_images.txt
#	-ex "break /home/local/KHQ/david.russell/dev/VIAME/src/packages/kwiver/sprokit/src/sprokit/pipeline/process.cxx:333" \
#	-ex "/home/local/KHQ/david.russell/dev/VIAME/src/packages/burn-out/library/object_detectors/conn_comp_super_process.txx:834" \
#       	-ex "break /home/local/KHQ/david.russell/dev/VIAME/src/packages/kwiver/arrows/burnout/burnout_detector.cxx:201" \
# \
