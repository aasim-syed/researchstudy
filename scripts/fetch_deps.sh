#!/usr/bin/env bash
# Fetch third-party dependencies (header-only libs + upstream QuadriFlow for the baseline).
set -euo pipefail
cd "$(dirname "$0")/../third_party"
clone() { [ -n "$(ls -A "$2" 2>/dev/null)" ] || { rm -rf "$2"; git clone -q --depth 1 ${3:+--branch $3} "$1" "$2"; }; }
clone https://github.com/eigen-mirror/eigen.git eigen
clone https://github.com/libigl/libigl.git libigl
clone https://github.com/pybind/pybind11.git pybind11
clone https://github.com/jkuhlmann/cgltf.git cgltf
clone https://github.com/hjwdzh/QuadriFlow.git QuadriFlow
clone https://github.com/wjakob/instant-meshes.git instant-meshes
for m in pcg32 dset pss tbb; do clone https://github.com/wjakob/$m instant-meshes/ext/$m; done
# baseline QuadriFlow CLI (unpatched)
if [ ! -x QuadriFlow/build/quadriflow ]; then
  cmake -S QuadriFlow -B QuadriFlow/build -DCMAKE_BUILD_TYPE=Release -DEIGEN_INCLUDE_DIR="$PWD/eigen" \
        -DEIGEN_INCLUDE_DIRS="$PWD/eigen" > /dev/null
  cmake --build QuadriFlow/build -j > /dev/null
fi
echo "dependencies ready"
