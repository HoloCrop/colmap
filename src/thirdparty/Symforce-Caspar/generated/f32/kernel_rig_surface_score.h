#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void RigSurfaceScore(float* pose,
                     unsigned int pose_num_alloc,
                     SharedIndex* pose_indices,
                     float* rig_point,
                     unsigned int rig_point_num_alloc,
                     float* position,
                     unsigned int position_num_alloc,
                     float* sqrt_weight,
                     unsigned int sqrt_weight_num_alloc,
                     const float* const robust,
                     float* const out_rTr,
                     size_t problem_size);

}  // namespace caspar