#include "kernel_rig_surface_score.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    RigSurfaceScoreKernel(float* pose,
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
                          size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27;

  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(
        rig_point, 0 * rig_point_num_alloc, global_thread_idx, r0, r1, r2);
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r3,
                       r4,
                       r5,
                       r6);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r7 = r3 * r3;
    r8 = -2.00000000000000000e+00;
    r7 = r7 * r8;
    r9 = 1.00000000000000000e+00;
    r10 = r4 * r4;
    r10 = fmaf(r8, r10, r9);
    r11 = r7 + r10;
    r12 = r4 * r6;
    r13 = 2.00000000000000000e+00;
    r14 = r3 * r13;
    r15 = r5 * r14;
    r12 = fmaf(r13, r12, r15);
    r16 = fmaf(r0, r12, r2 * r11);
    r17 = r4 * r5;
    r17 = r17 * r13;
    r18 = r6 * r8;
    r3 = fmaf(r3, r18, r17);
    r19 = -1.00000000000000000e+00;
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r20,
                       r21,
                       r22);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r21, r3, r22 * r11);
    r11 = fmaf(r20, r12, r11);
    ReadIdx3<1024, float, float, float4>(
        position, 0 * position_num_alloc, global_thread_idx, r12, r23, r24);
    r16 = fmaf(r1, r3, r16);
    r16 = fmaf(r19, r11, r16);
    r16 = fmaf(r24, r19, r16);
    r16 = r16 * r16;
  };
  LoadUnique<2, float, float>(robust, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared, 0, r24, r11);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r24 = r24 * r24;
    r3 = 0.00000000000000000e+00;
    r25 = 9.99999999999999955e-07;
    r25 = r25 + r16;
    r7 = r9 + r7;
    r9 = r5 * r5;
    r9 = r8 * r9;
    r7 = r7 + r9;
    r8 = r4 * r14;
    r26 = fmaf(r5, r18, r8);
    r27 = fmaf(r0, r26, r1 * r7);
    r14 = fmaf(r6, r14, r17);
    r7 = fmaf(r22, r14, r21 * r7);
    r7 = fmaf(r20, r26, r7);
    r27 = fmaf(r2, r14, r27);
    r27 = fmaf(r19, r7, r27);
    r27 = fmaf(r23, r19, r27);
    r27 = r27 * r27;
    r10 = r9 + r10;
    r18 = fmaf(r4, r18, r15);
    r2 = fmaf(r2, r18, r0 * r10);
    r0 = r5 * r6;
    r0 = fmaf(r13, r0, r8);
    r21 = fmaf(r21, r0, r20 * r10);
    r21 = fmaf(r22, r18, r21);
    r2 = fmaf(r1, r0, r2);
    r2 = fmaf(r19, r21, r2);
    r2 = fmaf(r12, r19, r2);
    r2 = r2 * r2;
    r25 = r25 + r27;
    r25 = r25 + r2;
    r12 = sqrtf(r25);
    r19 = fmaf(r11, r19, r12);
    r19 = fmaxf(r3, r19);
    r19 = r19 * r19;
    r3 = r24 + r19;
    r3 = 1.0 / r3;
    r25 = 1.0 / r25;
    r25 = r3 * r25;
    ReadIdx1<1024, float, float, float>(
        sqrt_weight, 0 * sqrt_weight_num_alloc, global_thread_idx, r3);
    r3 = r3 * r3;
    r25 = r25 * r24;
    r25 = r25 * r19;
    r25 = r25 * r3;
    r2 = fmaf(r25, r2, r25 * r16);
    r2 = fmaf(r25, r27, r2);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r2);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

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
                     size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  RigSurfaceScoreKernel<<<n_blocks, 1024>>>(pose,
                                            pose_num_alloc,
                                            pose_indices,
                                            rig_point,
                                            rig_point_num_alloc,
                                            position,
                                            position_num_alloc,
                                            sqrt_weight,
                                            sqrt_weight_num_alloc,
                                            robust,
                                            out_rTr,
                                            problem_size);
}

}  // namespace caspar