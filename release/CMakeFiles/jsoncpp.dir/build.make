# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

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
CMAKE_SOURCE_DIR = /home/ubuntu/Ludo

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ubuntu/Ludo/release

# Include any dependencies generated for this target.
include CMakeFiles/jsoncpp.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/jsoncpp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/jsoncpp.dir/flags.make

CMakeFiles/jsoncpp.dir/dist/jsoncpp.cpp.o: CMakeFiles/jsoncpp.dir/flags.make
CMakeFiles/jsoncpp.dir/dist/jsoncpp.cpp.o: ../dist/jsoncpp.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ubuntu/Ludo/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/jsoncpp.dir/dist/jsoncpp.cpp.o"
	/usr/bin/clang++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/jsoncpp.dir/dist/jsoncpp.cpp.o -c /home/ubuntu/Ludo/dist/jsoncpp.cpp

CMakeFiles/jsoncpp.dir/dist/jsoncpp.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/jsoncpp.dir/dist/jsoncpp.cpp.i"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ubuntu/Ludo/dist/jsoncpp.cpp > CMakeFiles/jsoncpp.dir/dist/jsoncpp.cpp.i

CMakeFiles/jsoncpp.dir/dist/jsoncpp.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/jsoncpp.dir/dist/jsoncpp.cpp.s"
	/usr/bin/clang++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ubuntu/Ludo/dist/jsoncpp.cpp -o CMakeFiles/jsoncpp.dir/dist/jsoncpp.cpp.s

# Object files for target jsoncpp
jsoncpp_OBJECTS = \
"CMakeFiles/jsoncpp.dir/dist/jsoncpp.cpp.o"

# External object files for target jsoncpp
jsoncpp_EXTERNAL_OBJECTS =

libjsoncpp.a: CMakeFiles/jsoncpp.dir/dist/jsoncpp.cpp.o
libjsoncpp.a: CMakeFiles/jsoncpp.dir/build.make
libjsoncpp.a: CMakeFiles/jsoncpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ubuntu/Ludo/release/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX static library libjsoncpp.a"
	$(CMAKE_COMMAND) -P CMakeFiles/jsoncpp.dir/cmake_clean_target.cmake
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/jsoncpp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/jsoncpp.dir/build: libjsoncpp.a

.PHONY : CMakeFiles/jsoncpp.dir/build

CMakeFiles/jsoncpp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/jsoncpp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/jsoncpp.dir/clean

CMakeFiles/jsoncpp.dir/depend:
	cd /home/ubuntu/Ludo/release && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ubuntu/Ludo /home/ubuntu/Ludo /home/ubuntu/Ludo/release /home/ubuntu/Ludo/release /home/ubuntu/Ludo/release/CMakeFiles/jsoncpp.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/jsoncpp.dir/depend

