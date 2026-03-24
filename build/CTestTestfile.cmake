# CMake generated Testfile for 
# Source directory: C:/Users/j.kalnius/Documents/Process Simulator
# Build directory: C:/Users/j.kalnius/Documents/Process Simulator/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[all_tests]=] "C:/msys64/ucrt64/bin/cmake.exe" "-E" "env" "PATH=C:/Users/j.kalnius/Documents/Process Simulator/build;C:/msys64/mingw64/bin;C:/msys64/ucrt64/bin;C:\\msys64\\mingw64\\bin;C:\\msys64\\mingw64\\bin;C:\\msys64\\mingw64\\bin;C:\\msys64\\usr\\bin;C:/Users/j.kalnius/AppData/Local/Programs/Microsoft VS Code;C:\\Windows\\system32;C:/Windows;C:\\Windows\\System32\\Wbem;C:/Windows/System32/WindowsPowerShell/v1.0/;C:\\Windows\\System32\\OpenSSH\\;C:/Program Files/dotnet/;C:\\Program Files\\Microsoft SQL Server\\150\\Tools\\Binn\\;C:/Siemens/Teamcenter/Tc13/Visualization/Products/Mockup/ClearanceDB;C:\\Program Files\\Microsoft SQL Server\\120\\Tools\\Binn\\;C:/Program Files/Git/cmd;C:\\Users\\j.kalnius\\AppData\\Local\\Programs\\Python\\Python314\\Scripts\\;C:/Users/j.kalnius/AppData/Local/Programs/Python/Python314/;C:\\Users\\j.kalnius\\AppData\\Local\\Microsoft\\WindowsApps;C:/Users/j.kalnius/AppData/Local/Programs/Microsoft VS Code/bin;C:\\Program Files\\Git\\bin;C:/Users/j.kalnius/AppData/Local/Programs/cursor/resources/app/bin;C:\\Users\\j.kalnius\\AppData\\Local\\Programs\\Ollama;C:/msys64/ucrt64/bin;" "C:/Users/j.kalnius/Documents/Process Simulator/build/run_tests.exe")
set_tests_properties([=[all_tests]=] PROPERTIES  _BACKTRACE_TRIPLES "C:/Users/j.kalnius/Documents/Process Simulator/CMakeLists.txt;64;add_test;C:/Users/j.kalnius/Documents/Process Simulator/CMakeLists.txt;0;")
subdirs("_deps/googletest-build")
subdirs("_deps/nlohmann_json-build")
subdirs("_deps/pybind11-build")
