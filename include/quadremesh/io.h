#pragma once
#include <string>

#include "quadremesh/types.h"

namespace qrm {

// Load .obj / .ply / .glb / .gltf / .off. Polygons are fan-triangulated; glTF scenes are
// flattened with node transforms applied. Throws std::runtime_error on failure.
TriMesh load_mesh(const std::string& path);

// Write a quad-dominant mesh as OBJ (quads + triangles), with per-corner UVs if present.
void save_quad_obj(const std::string& path, const QuadMesh& m);
void save_tri_obj(const std::string& path, const TriMesh& m);
// Load a polygon OBJ as a quad-dominant mesh (n-gons with n>4 are fan-split into triangles).
QuadMesh load_quad_obj(const std::string& path);

std::string lower_ext(const std::string& path);

}  // namespace qrm
