#pragma once
// Quadric-error edge-collapse decimation (Garland & Heckbert 1997) with link-condition and
// normal-flip checks; open boundaries are preserved through boundary-plane quadrics.
// Used for the last-resort quad fallback so its face count still follows the target.
#include "quadremesh/types.h"

namespace qrm {

void decimate_qem(MatrixXd& V, MatrixXi& F, int target_faces);

}  // namespace qrm
