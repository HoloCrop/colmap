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
    RigSurfaceScoreKernel(double* pose,
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
                          size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  __shared__ double out_rTr_local[1];

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24;

  if (global_thread_idx < problem_size) {
    ReadIdx2<1024, double, double, double2>(
        rig_point, 0 * rig_point_num_alloc, global_thread_idx, r0, r1);
    r2 = -2.00000000000000000e+00;
  };
  LoadShared<2, double, double>(
      pose, 2 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r3, r4);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r5 = r3 * r3;
    r5 = r2 * r5;
    r6 = 1.00000000000000000e+00;
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r7, r8);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r9 = r8 * r8;
    r9 = fma(r2, r9, r6);
    r10 = r5 + r9;
    ReadIdx1<1024, double, double, double>(
        rig_point, 2 * rig_point_num_alloc, global_thread_idx, r11);
    r12 = 2.00000000000000000e+00;
    r13 = r7 * r12;
    r14 = r3 * r13;
    r15 = r4 * r2;
    r16 = fma(r8, r15, r14);
    r17 = fma(r11, r16, r0 * r10);
    r18 = r3 * r4;
    r19 = r8 * r13;
    r18 = fma(r12, r18, r19);
    r20 = -1.00000000000000000e+00;
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r21, r22);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = fma(r22, r18, r21 * r10);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r23);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = fma(r23, r16, r10);
    ReadIdx2<1024, double, double, double2>(
        position, 0 * position_num_alloc, global_thread_idx, r16, r24);
    r17 = fma(r1, r18, r17);
    r17 = fma(r20, r10, r17);
    r17 = fma(r16, r20, r17);
    r17 = r17 * r17;
    r16 = 1.00000000000000008e-15;
    r16 = r16 + r17;
    r10 = r7 * r7;
    r10 = r10 * r2;
    r9 = r10 + r9;
    r2 = r8 * r4;
    r2 = fma(r12, r2, r14);
    r14 = fma(r0, r2, r11 * r9);
    r18 = r8 * r3;
    r18 = r18 * r12;
    r7 = fma(r7, r15, r18);
    r9 = fma(r22, r7, r23 * r9);
    r9 = fma(r21, r2, r9);
    ReadIdx1<1024, double, double, double>(
        position, 2 * position_num_alloc, global_thread_idx, r2);
    r14 = fma(r1, r7, r14);
    r14 = fma(r20, r9, r14);
    r14 = fma(r2, r20, r14);
    r14 = r14 * r14;
    r5 = r6 + r5;
    r5 = r5 + r10;
    r15 = fma(r3, r15, r19);
    r0 = fma(r0, r15, r1 * r5);
    r13 = fma(r4, r13, r18);
    r23 = fma(r23, r13, r22 * r5);
    r23 = fma(r21, r15, r23);
    r0 = fma(r11, r13, r0);
    r0 = fma(r20, r23, r0);
    r0 = fma(r24, r20, r0);
    r0 = r0 * r0;
    r16 = r16 + r14;
    r16 = r16 + r0;
    r24 = 1.0 / r16;
  };
  LoadUnique<2, double, double>(robust, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared, 0, r23, r13);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r23 = r23 * r23;
    r11 = 0.00000000000000000e+00;
    r16 = sqrt(r16);
    r20 = fma(r13, r20, r16);
    r20 = fmax(r11, r20);
    r20 = r20 * r20;
    r11 = r23 + r20;
    r11 = 1.0 / r11;
    r11 = r24 * r11;
    ReadIdx1<1024, double, double, double>(
        sqrt_weight, 0 * sqrt_weight_num_alloc, global_thread_idx, r24);
    r24 = r24 * r24;
    r11 = r11 * r23;
    r11 = r11 * r20;
    r11 = r11 * r24;
    r14 = fma(r11, r14, r11 * r17);
    r14 = fma(r11, r0, r14);
  };
  SumStore<double>(out_rTr_local,
                   (double*)inout_shared,
                   0,
                   global_thread_idx < problem_size,
                   r14);
  SumFlushFinal<double>(out_rTr_local, out_rTr, 1);
}

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