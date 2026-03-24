cmake_minimum_required(VERSION 3.20)

if(APPLE AND "${CMAKE_OSX_ARCHITECTURES}" STREQUAL "")
  execute_process(
    COMMAND uname -m
    OUTPUT_VARIABLE CHEMSIM_HOST_ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE)
  if(CHEMSIM_HOST_ARCH STREQUAL "arm64")
    set(CHEMSIM_FORCE_ARM64 ON)
    set(CMAKE_APPLE_SILICON_PROCESSOR "arm64" CACHE STRING "Apple Silicon processor" FORCE)
    set(CMAKE_OSX_ARCHITECTURES "arm64" CACHE STRING "Build architectures for macOS" FORCE)
  endif()
endif()

project(ChemSim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(APPLE AND CHEMSIM_FORCE_ARM64)
  add_compile_options(-arch arm64)
  add_link_options(-arch arm64)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

# ── Dependencies ──────────────────────────────────────────────────────────────
include(GNUInstallDirs)
include(CTest)

find_package(Eigen3 REQUIRED)

option(BUILD_PYTHON "Build the pybind11 Python extension" ON)
option(CHEMSIM_USE_SYSTEM_DEPS "Prefer locally installed packages before FetchContent" ON)

option(BUILD_GUI "Build the Qt6 GUI application" OFF)
if(BUILD_GUI)
  find_package(Qt6 COMPONENTS Widgets Svg QUIET)
endif()

include(FetchContent)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)

if(CHEMSIM_USE_SYSTEM_DEPS)
  find_package(nlohmann_json CONFIG QUIET)
endif()
if(NOT nlohmann_json_FOUND)
  FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
  )
  FetchContent_MakeAvailable(nlohmann_json)
endif()

if(BUILD_PYTHON)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)
  if(CHEMSIM_USE_SYSTEM_DEPS)
    find_package(pybind11 CONFIG QUIET)
  endif()

  if(NOT pybind11_FOUND)
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.13.6
    )
    FetchContent_MakeAvailable(pybind11)
  endif()
endif()

# ── Core library ──────────────────────────────────────────────────────────────
file(GLOB_RECURSE CHEMSIM_SOURCES "src/**/*.cpp")
list(FILTER CHEMSIM_SOURCES EXCLUDE REGEX ".*/src/gui/.*\\.cpp$")
add_library(chemsim_core STATIC ${CHEMSIM_SOURCES})
target_include_directories(chemsim_core PUBLIC include)
target_link_libraries(chemsim_core PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json)

if(BUILD_PYTHON)
  pybind11_add_module(chemsim MODULE python/chemsim.cpp)
  target_link_libraries(chemsim PRIVATE chemsim_core)
  target_compile_definitions(chemsim PRIVATE
      CHEMSIM_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
endif()

if(BUILD_GUI AND Qt6_FOUND)
  add_executable(chemsim_gui
    src/gui/main.cpp
    src/gui/MainWindow.cpp
    src/gui/FlowsheetCanvas.cpp)
  target_include_directories(chemsim_gui PRIVATE src/gui)
  target_link_libraries(chemsim_gui PRIVATE chemsim_core Qt6::Widgets Qt6::Svg)
  target_compile_definitions(chemsim_gui PRIVATE
      CHEMSIM_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
elseif(BUILD_GUI AND NOT Qt6_FOUND)
  message(WARNING "BUILD_GUI=ON but Qt6 was not found — GUI target skipped")
endif()

# ── Tests ─────────────────────────────────────────────────────────────────────
if(BUILD_TESTING)
  if(CHEMSIM_USE_SYSTEM_DEPS)
    find_package(GTest CONFIG QUIET)
  endif()

  if(NOT GTest_FOUND)
    FetchContent_Declare(
      googletest
      GIT_REPOSITORY https://github.com/google/googletest.git
      GIT_TAG        v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
  endif()

  file(GLOB TEST_SOURCES "tests/*.cpp")
  add_executable(run_tests ${TEST_SOURCES})
  target_link_libraries(run_tests chemsim_core GTest::gtest_main)
  target_compile_definitions(run_tests PRIVATE
      CHEMSIM_DATA_DIR="${CMAKE_SOURCE_DIR}/data")

  if(WIN32)
    add_test(
      NAME all_tests
      COMMAND ${CMAKE_COMMAND} -E env
              "PATH=$<TARGET_FILE_DIR:run_tests>;C:/msys64/mingw64/bin;C:/msys64/ucrt64/bin;$ENV{PATH}"
              $<TARGET_FILE:run_tests>)
  else()
    add_test(NAME all_tests COMMAND run_tests)
  endif()

  set_tests_properties(all_tests PROPERTIES
    WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
endif()

# ── Install ───────────────────────────────────────────────────────────────────
install(TARGETS chemsim_core
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

if(BUILD_PYTHON)
  install(TARGETS chemsim
    LIBRARY DESTINATION chemsim
    RUNTIME DESTINATION chemsim)
endif()
