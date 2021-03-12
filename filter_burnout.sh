gdb -ex "set breakpoint pending on" \
	-ex "break /home/local/KHQ/david.russell/dev/VIAME/src/packages/kwiver/arrows/vxl/image_io.cxx:552" \
	-ex run --args kwiver runner filter_burnout.pipe -s input:video_filename=/home/local/KHQ/david.russell/data/NOAA/image_list.txt
