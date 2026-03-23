cmake_minimum_required(VERSION 3.20)
project(ChemSim VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -Wall -Wextra")

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(Eigen3 REQUIRED)

# GoogleTest via FetchContent (no git submodule needed for now)
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# nlohmann/json via FetchContent
FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG        v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# ── Core library ──────────────────────────────────────────────────────────────
file(GLOB_RECURSE CHEMSIM_SOURCES "src/**/*.cpp")
add_library(chemsim_core STATIC ${CHEMSIM_SOURCES})
target_include_directories(chemsim_core PUBLIC include)
target_link_libraries(chemsim_core PUBLIC Eigen3::Eigen nlohmann_json::nlohmann_json)

# ── Tests ─────────────────────────────────────────────────────────────────────
enable_testing()
file(GLOB TEST_SOURCES "tests/*.cpp")
add_executable(run_tests ${TEST_SOURCES})
target_link_libraries(run_tests chemsim_core GTest::gtest_main)
target_compile_definitions(run_tests PRIVATE
    CHEMSIM_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
add_test(NAME all_tests COMMAND run_tests)
