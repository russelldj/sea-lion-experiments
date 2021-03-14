#gdb -ex "set breakpoint pending on" \
#	-ex run --args
kwiver runner detector_burnout.pipe -s input:video_filename=input_images.txt
