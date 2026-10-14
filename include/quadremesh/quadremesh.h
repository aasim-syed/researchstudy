#pragma once
// Public API.
#include "quadremesh/io.h"
#include "quadremesh/metrics.h"
#include "quadremesh/types.h"

namespace qrm {

// Convert a triangle mesh into a structured quad (or quad-dominant) mesh.
// Deterministic for identical input and parameters.
Result quadremesh(const TriMesh& input, const Params& params);

const char* version();

}  // namespace qrm
