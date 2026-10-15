#pragma once

#include "shared_indices.h"
#include <cuda_runtime.h>

namespace caspar {

void RigSurfaceResJacFirst(double* pose,
                           unsigned int pose_num_alloc,
                           SharedIndex* pose_indices,
                           double* rig_point,
                           unsigned int rig_point_num_alloc,
                           double* position,
                           unsigned int position_num_alloc,
                           double* sqrt_weight,
                           unsigned int sqrt_weight_num_alloc,
                           const double* const robust,
                           double* out_res,
                           unsigned int out_res_num_alloc,
                           double* const out_rTr,
                           double* const out_pose_njtr,
                           unsigned int out_pose_njtr_num_alloc,
                           double* const out_pose_precond_diag,
                           unsigned int out_pose_precond_diag_num_alloc,
                           double* const out_pose_precond_tril,
                           unsigned int out_pose_precond_tril_num_alloc,
                           size_t problem_size);

}  // namespace caspar