# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
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
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/kang/Project/okvis

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/kang/Project/okvis/build

# Include any dependencies generated for this target.
include okvis_util/CMakeFiles/okvis_util.dir/depend.make

# Include the progress variables for this target.
include okvis_util/CMakeFiles/okvis_util.dir/progress.make

# Include the compile flags for this target's objects.
include okvis_util/CMakeFiles/okvis_util.dir/flags.make

okvis_util/CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.o: okvis_util/CMakeFiles/okvis_util.dir/flags.make
okvis_util/CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.o: ../okvis_util/src/dependency-tracker.cc
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/kang/Project/okvis/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object okvis_util/CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.o"
	cd /home/kang/Project/okvis/build/okvis_util && /usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.o -c /home/kang/Project/okvis/okvis_util/src/dependency-tracker.cc

okvis_util/CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.i"
	cd /home/kang/Project/okvis/build/okvis_util && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/kang/Project/okvis/okvis_util/src/dependency-tracker.cc > CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.i

okvis_util/CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.s"
	cd /home/kang/Project/okvis/build/okvis_util && /usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/kang/Project/okvis/okvis_util/src/dependency-tracker.cc -o CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.s

okvis_util/CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.o.requires:

.PHONY : okvis_util/CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.o.requires

okvis_util/CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.o.provides: okvis_util/CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.o.requires
	$(MAKE) -f okvis_util/CMakeFiles/okvis_util.dir/build.make okvis_util/CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.o.provides.build
.PHONY : okvis_util/CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.o.provides

okvis_util/CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.o.provides.build: okvis_util/CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.o


# Object files for target okvis_util
okvis_util_OBJECTS = \
"CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.o"

# External object files for target okvis_util
okvis_util_EXTERNAL_OBJECTS =

okvis_util/libokvis_util.a: okvis_util/CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.o
okvis_util/libokvis_util.a: okvis_util/CMakeFiles/okvis_util.dir/build.make
okvis_util/libokvis_util.a: okvis_util/CMakeFiles/okvis_util.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/kang/Project/okvis/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libokvis_util.a"
	cd /home/kang/Project/okvis/build/okvis_util && $(CMAKE_COMMAND) -P CMakeFiles/okvis_util.dir/cmake_clean_target.cmake
	cd /home/kang/Project/okvis/build/okvis_util && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/okvis_util.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
okvis_util/CMakeFiles/okvis_util.dir/build: okvis_util/libokvis_util.a

.PHONY : okvis_util/CMakeFiles/okvis_util.dir/build

okvis_util/CMakeFiles/okvis_util.dir/requires: okvis_util/CMakeFiles/okvis_util.dir/src/dependency-tracker.cc.o.requires

.PHONY : okvis_util/CMakeFiles/okvis_util.dir/requires

okvis_util/CMakeFiles/okvis_util.dir/clean:
	cd /home/kang/Project/okvis/build/okvis_util && $(CMAKE_COMMAND) -P CMakeFiles/okvis_util.dir/cmake_clean.cmake
.PHONY : okvis_util/CMakeFiles/okvis_util.dir/clean

okvis_util/CMakeFiles/okvis_util.dir/depend:
	cd /home/kang/Project/okvis/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/kang/Project/okvis /home/kang/Project/okvis/okvis_util /home/kang/Project/okvis/build /home/kang/Project/okvis/build/okvis_util /home/kang/Project/okvis/build/okvis_util/CMakeFiles/okvis_util.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : okvis_util/CMakeFiles/okvis_util.dir/depend

