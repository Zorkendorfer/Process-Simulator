# ChemSim

ChemSim is a steady-state C++ process simulator with Peng-Robinson thermodynamics, unit operations, recycle-capable flowsheets, JSON I/O, Python bindings, and a Gymnasium environment for RL experiments.

## Features

- Peng-Robinson EOS with TP, PH, and PS flash support
- Unit operations for flash drums, pumps, compressors, heat exchangers, reactors, and distillation
- Flowsheet graph with recycle detection and direct-substitution convergence
- JSON flowsheet parsing and JSON result export
- Python bindings for loading, solving, and inspecting flowsheets
- Gymnasium environment for distillation-control experiments
- GoogleTest coverage across thermo, unit ops, flowsheet solving, and polish APIs

## macOS Quick Start

### 1. Install toolchain

Homebrew is the easiest path on macOS:

```bash
brew install cmake ninja eigen python
```

If you want the desktop GUI as well:

```bash
brew install qt
```

You also need Apple command line tools:

```bash
xcode-select --install
```

### 2. Build and test the C++ project

```bash
cmake -S . -B build-macos -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DBUILD_PYTHON=ON \
  -DBUILD_TESTING=ON

cmake --build build-macos
ctest --test-dir build-macos --output-on-failure
```

To build the optional Qt GUI:

```bash
cmake -S . -B build-macos -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DBUILD_GUI=ON \
  -DBUILD_PYTHON=ON \
  -DBUILD_TESTING=ON
```

### 3. Install the Python package in editable mode

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=arm64 -DBUILD_TESTING=OFF" python -m pip install -e .
```

This installs:

- `chemsim` — the pybind11 extension plus Python package
- `chemsim_gym` — the Gymnasium environment wrapper

## Windows Quick Start

### MinGW64 / MSYS2

```powershell
cd "C:\Users\j.kalnius\Documents\Process Simulator"
$env:PATH="C:\msys64\mingw64\bin;C:\msys64\usr\bin;$env:PATH"

C:\msys64\mingw64\bin\cmake.exe -S . -B build-mingw64 -G Ninja `
  -DCMAKE_C_COMPILER=C:\msys64\mingw64\bin\gcc.exe `
  -DCMAKE_CXX_COMPILER=C:\msys64\mingw64\bin\g++.exe `
  -DBUILD_PYTHON=ON `
  -DBUILD_TESTING=ON `
  -DFETCHCONTENT_UPDATES_DISCONNECTED=ON

C:\msys64\mingw64\bin\ninja.exe -C build-mingw64 run_tests chemsim
.\build-mingw64\run_tests.exe
```

## Python Example

```python
import chemsim

fs = chemsim.Flowsheet.from_json("examples/simple_recycle.json", "data/components.json")
fs.solve()
print(fs.summary())
print(fs.get_stream("PRODUCT").total_flow)
fs.export_results("results.json")
```

## Notes For Local Development

- `chemsim_gym` can import the extension from an installed package or from local build directories such as `build/` or `build-macos/`.
- Set `CHEMSIM_BUILD_DIR=/path/to/build-dir` if your extension is in a non-standard build folder.
- The CMake project now supports `BUILD_PYTHON`, `BUILD_TESTING`, and `BUILD_GUI` toggles so you can bring up the macOS toolchain incrementally.
- On Apple Silicon, pass `-DCMAKE_OSX_ARCHITECTURES=arm64` explicitly if your local CMake toolchain defaults to `x86_64`.

## Example Flowsheets

See `examples/simple_recycle.json` for a small recycle loop example and `examples/distillation_recycle.json` for the RL distillation setup.
