# CMake generated Testfile for 
# Source directory: C:/Users/j.kalnius/Documents/Process Simulator
# Build directory: C:/Users/j.kalnius/Documents/Process Simulator/build2
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[all_tests]=] "C:/Users/j.kalnius/Documents/Process Simulator/build2/run_tests.exe")
set_tests_properties([=[all_tests]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/j.kalnius/Documents/Process Simulator/CMakeLists.txt;48;add_test;C:/Users/j.kalnius/Documents/Process Simulator/CMakeLists.txt;0;")
subdirs("_deps/googletest-build")
subdirs("_deps/nlohmann_json-build")
