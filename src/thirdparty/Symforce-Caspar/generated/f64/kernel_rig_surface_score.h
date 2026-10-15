#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void RigSurfaceScore(double* pose,
                     unsigned int pose_num_alloc,
                     SharedIndex* pose_indices,
                     double* rig_point,
                     unsigned int rig_point_num_alloc,
                     double* position,
                     unsigned int position_num_alloc,
                     double* sqrt_weight,
                     unsigned int sqrt_weight_num_alloc,
                     const double* const robust,
                     double* const out_rTr,
                     size_t problem_size);

}  // namespace caspar