# ChemSim

ChemSim is a steady-state C++ process simulator with Peng-Robinson thermodynamics, unit operations, recycle-capable flowsheets, JSON I/O, and Python bindings.

## Features

- Peng-Robinson EOS with TP, PH, and PS flash support
- Unit operations for flash drums, pumps, compressors, heat exchangers, reactors, and distillation
- Flowsheet graph with recycle detection and direct-substitution convergence
- JSON flowsheet parsing and JSON result export
- Python bindings for loading, solving, and inspecting flowsheets
- GoogleTest coverage across thermo, unit ops, flowsheet solving, and polish APIs

## Build

### Windows MinGW64

```powershell
cd "C:\Users\j.kalnius\Documents\Process Simulator"
$env:PATH="C:\msys64\mingw64\bin;C:\msys64\usr\bin;$env:PATH"

C:\msys64\mingw64\bin\cmake.exe -S . -B build-mingw64 -G Ninja `
  -DCMAKE_C_COMPILER=C:\msys64\mingw64\bin\gcc.exe `
  -DCMAKE_CXX_COMPILER=C:\msys64\mingw64\bin\g++.exe `
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

## JSON Example

See [`examples/simple_recycle.json`](c:/Users/j.kalnius/Documents/Process%20Simulator/examples/simple_recycle.json) for a small recycle loop example.
