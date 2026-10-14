#pragma once

#include "colmap/geometry/pose_prior.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/hash_containers.h"

#include <vector>

namespace colmap {

// Pose priors of rig frames. A prior belongs to a frame when its corr_data_id
// is the frame's data for the rig's reference sensor, so its position is the
// rig origin in world. The reference sensor is either a camera (the prior is
// on that camera's image) or a non-camera sensor at the rig origin, such as a
// GNSS/INS unit, whose data id is part of every frame. Priors on other sensors
// are ignored: each frame has one rig pose and therefore one prior.
NodeHashMap<frame_t, const PosePrior*> FramePosePriors(
    const Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors);

}  // namespace colmap
