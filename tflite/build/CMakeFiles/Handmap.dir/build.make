# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.25

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/junghwalee/hand/tflite

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/junghwalee/hand/tflite/build

# Include any dependencies generated for this target.
include CMakeFiles/Handmap.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/Handmap.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/Handmap.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/Handmap.dir/flags.make

CMakeFiles/Handmap.dir/src/main.cpp.o: CMakeFiles/Handmap.dir/flags.make
CMakeFiles/Handmap.dir/src/main.cpp.o: /home/junghwalee/hand/tflite/src/main.cpp
CMakeFiles/Handmap.dir/src/main.cpp.o: CMakeFiles/Handmap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/junghwalee/hand/tflite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/Handmap.dir/src/main.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Handmap.dir/src/main.cpp.o -MF CMakeFiles/Handmap.dir/src/main.cpp.o.d -o CMakeFiles/Handmap.dir/src/main.cpp.o -c /home/junghwalee/hand/tflite/src/main.cpp

CMakeFiles/Handmap.dir/src/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Handmap.dir/src/main.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/junghwalee/hand/tflite/src/main.cpp > CMakeFiles/Handmap.dir/src/main.cpp.i

CMakeFiles/Handmap.dir/src/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Handmap.dir/src/main.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/junghwalee/hand/tflite/src/main.cpp -o CMakeFiles/Handmap.dir/src/main.cpp.s

CMakeFiles/Handmap.dir/src/ModelLoader.cpp.o: CMakeFiles/Handmap.dir/flags.make
CMakeFiles/Handmap.dir/src/ModelLoader.cpp.o: /home/junghwalee/hand/tflite/src/ModelLoader.cpp
CMakeFiles/Handmap.dir/src/ModelLoader.cpp.o: CMakeFiles/Handmap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/junghwalee/hand/tflite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/Handmap.dir/src/ModelLoader.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Handmap.dir/src/ModelLoader.cpp.o -MF CMakeFiles/Handmap.dir/src/ModelLoader.cpp.o.d -o CMakeFiles/Handmap.dir/src/ModelLoader.cpp.o -c /home/junghwalee/hand/tflite/src/ModelLoader.cpp

CMakeFiles/Handmap.dir/src/ModelLoader.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Handmap.dir/src/ModelLoader.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/junghwalee/hand/tflite/src/ModelLoader.cpp > CMakeFiles/Handmap.dir/src/ModelLoader.cpp.i

CMakeFiles/Handmap.dir/src/ModelLoader.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Handmap.dir/src/ModelLoader.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/junghwalee/hand/tflite/src/ModelLoader.cpp -o CMakeFiles/Handmap.dir/src/ModelLoader.cpp.s

CMakeFiles/Handmap.dir/src/DetectionPostProcess.cpp.o: CMakeFiles/Handmap.dir/flags.make
CMakeFiles/Handmap.dir/src/DetectionPostProcess.cpp.o: /home/junghwalee/hand/tflite/src/DetectionPostProcess.cpp
CMakeFiles/Handmap.dir/src/DetectionPostProcess.cpp.o: CMakeFiles/Handmap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/junghwalee/hand/tflite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object CMakeFiles/Handmap.dir/src/DetectionPostProcess.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Handmap.dir/src/DetectionPostProcess.cpp.o -MF CMakeFiles/Handmap.dir/src/DetectionPostProcess.cpp.o.d -o CMakeFiles/Handmap.dir/src/DetectionPostProcess.cpp.o -c /home/junghwalee/hand/tflite/src/DetectionPostProcess.cpp

CMakeFiles/Handmap.dir/src/DetectionPostProcess.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Handmap.dir/src/DetectionPostProcess.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/junghwalee/hand/tflite/src/DetectionPostProcess.cpp > CMakeFiles/Handmap.dir/src/DetectionPostProcess.cpp.i

CMakeFiles/Handmap.dir/src/DetectionPostProcess.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Handmap.dir/src/DetectionPostProcess.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/junghwalee/hand/tflite/src/DetectionPostProcess.cpp -o CMakeFiles/Handmap.dir/src/DetectionPostProcess.cpp.s

CMakeFiles/Handmap.dir/src/Handlandmark.cpp.o: CMakeFiles/Handmap.dir/flags.make
CMakeFiles/Handmap.dir/src/Handlandmark.cpp.o: /home/junghwalee/hand/tflite/src/Handlandmark.cpp
CMakeFiles/Handmap.dir/src/Handlandmark.cpp.o: CMakeFiles/Handmap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/junghwalee/hand/tflite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Building CXX object CMakeFiles/Handmap.dir/src/Handlandmark.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Handmap.dir/src/Handlandmark.cpp.o -MF CMakeFiles/Handmap.dir/src/Handlandmark.cpp.o.d -o CMakeFiles/Handmap.dir/src/Handlandmark.cpp.o -c /home/junghwalee/hand/tflite/src/Handlandmark.cpp

CMakeFiles/Handmap.dir/src/Handlandmark.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Handmap.dir/src/Handlandmark.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/junghwalee/hand/tflite/src/Handlandmark.cpp > CMakeFiles/Handmap.dir/src/Handlandmark.cpp.i

CMakeFiles/Handmap.dir/src/Handlandmark.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Handmap.dir/src/Handlandmark.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/junghwalee/hand/tflite/src/Handlandmark.cpp -o CMakeFiles/Handmap.dir/src/Handlandmark.cpp.s

CMakeFiles/Handmap.dir/src/HandDetection.cpp.o: CMakeFiles/Handmap.dir/flags.make
CMakeFiles/Handmap.dir/src/HandDetection.cpp.o: /home/junghwalee/hand/tflite/src/HandDetection.cpp
CMakeFiles/Handmap.dir/src/HandDetection.cpp.o: CMakeFiles/Handmap.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/junghwalee/hand/tflite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_5) "Building CXX object CMakeFiles/Handmap.dir/src/HandDetection.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/Handmap.dir/src/HandDetection.cpp.o -MF CMakeFiles/Handmap.dir/src/HandDetection.cpp.o.d -o CMakeFiles/Handmap.dir/src/HandDetection.cpp.o -c /home/junghwalee/hand/tflite/src/HandDetection.cpp

CMakeFiles/Handmap.dir/src/HandDetection.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/Handmap.dir/src/HandDetection.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/junghwalee/hand/tflite/src/HandDetection.cpp > CMakeFiles/Handmap.dir/src/HandDetection.cpp.i

CMakeFiles/Handmap.dir/src/HandDetection.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/Handmap.dir/src/HandDetection.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/junghwalee/hand/tflite/src/HandDetection.cpp -o CMakeFiles/Handmap.dir/src/HandDetection.cpp.s

# Object files for target Handmap
Handmap_OBJECTS = \
"CMakeFiles/Handmap.dir/src/main.cpp.o" \
"CMakeFiles/Handmap.dir/src/ModelLoader.cpp.o" \
"CMakeFiles/Handmap.dir/src/DetectionPostProcess.cpp.o" \
"CMakeFiles/Handmap.dir/src/Handlandmark.cpp.o" \
"CMakeFiles/Handmap.dir/src/HandDetection.cpp.o"

# External object files for target Handmap
Handmap_EXTERNAL_OBJECTS =

Handmap: CMakeFiles/Handmap.dir/src/main.cpp.o
Handmap: CMakeFiles/Handmap.dir/src/ModelLoader.cpp.o
Handmap: CMakeFiles/Handmap.dir/src/DetectionPostProcess.cpp.o
Handmap: CMakeFiles/Handmap.dir/src/Handlandmark.cpp.o
Handmap: CMakeFiles/Handmap.dir/src/HandDetection.cpp.o
Handmap: CMakeFiles/Handmap.dir/build.make
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_stitching.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_alphamat.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_aruco.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_barcode.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_bgsegm.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_bioinspired.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_ccalib.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_cvv.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_dnn_objdetect.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_dnn_superres.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_dpm.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_face.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_freetype.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_fuzzy.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_hdf.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_hfs.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_img_hash.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_intensity_transform.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_line_descriptor.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_mcc.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_quality.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_rapid.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_reg.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_rgbd.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_saliency.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_shape.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_stereo.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_structured_light.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_superres.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_surface_matching.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_tracking.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_videostab.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_viz.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_wechat_qrcode.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_xobjdetect.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_xphoto.so.4.6.0
Handmap: /home/junghwalee/hand/tflite/lib/libtensorflowlite.so
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_highgui.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_datasets.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_plot.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_text.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_ml.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_phase_unwrapping.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_optflow.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_ximgproc.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_video.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_videoio.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_imgcodecs.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_objdetect.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_calib3d.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_dnn.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_features2d.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_flann.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_photo.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_imgproc.so.4.6.0
Handmap: /usr/lib/aarch64-linux-gnu/libopencv_core.so.4.6.0
Handmap: CMakeFiles/Handmap.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/junghwalee/hand/tflite/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_6) "Linking CXX executable Handmap"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/Handmap.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/Handmap.dir/build: Handmap
.PHONY : CMakeFiles/Handmap.dir/build

CMakeFiles/Handmap.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/Handmap.dir/cmake_clean.cmake
.PHONY : CMakeFiles/Handmap.dir/clean

CMakeFiles/Handmap.dir/depend:
	cd /home/junghwalee/hand/tflite/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/junghwalee/hand/tflite /home/junghwalee/hand/tflite /home/junghwalee/hand/tflite/build /home/junghwalee/hand/tflite/build /home/junghwalee/hand/tflite/build/CMakeFiles/Handmap.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/Handmap.dir/depend

