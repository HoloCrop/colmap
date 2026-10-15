#include "kernel_rig_surface_res_jac.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    RigSurfaceResJacKernel(float* pose,
                           unsigned int pose_num_alloc,
                           SharedIndex* pose_indices,
                           float* rig_point,
                           unsigned int rig_point_num_alloc,
                           float* position,
                           unsigned int position_num_alloc,
                           float* sqrt_weight,
                           unsigned int sqrt_weight_num_alloc,
                           const float* const robust,
                           float* out_res,
                           unsigned int out_res_num_alloc,
                           float* const out_pose_njtr,
                           unsigned int out_pose_njtr_num_alloc,
                           float* const out_pose_precond_diag,
                           unsigned int out_pose_precond_diag_num_alloc,
                           float* const out_pose_precond_tril,
                           unsigned int out_pose_precond_tril_num_alloc,
                           size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54;

  if (global_thread_idx < problem_size) {
    ReadIdx3<1024, float, float, float4>(
        rig_point, 0 * rig_point_num_alloc, global_thread_idx, r0, r1, r2);
    r3 = -2.00000000000000000e+00;
  };
  LoadShared<4, float, float>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared4<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r4,
                       r5,
                       r6,
                       r7);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r8 = r6 * r6;
    r9 = r3 * r8;
    r10 = 1.00000000000000000e+00;
    r11 = r5 * r5;
    r12 = fmaf(r3, r11, r10);
    r13 = r9 + r12;
    r14 = -1.00000000000000000e+00;
  };
  LoadShared<3, float, float>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared3<float>((float*)inout_shared,
                       pose_indices_loc[threadIdx.x].target,
                       r15,
                       r16,
                       r17);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r18 = r6 * r7;
    r19 = 2.00000000000000000e+00;
    r18 = r18 * r19;
    r20 = r4 * r19;
    r21 = r5 * r20;
    r22 = r18 + r21;
    r23 = r16 * r22;
    r24 = fmaf(r15, r13, r23);
    r25 = r6 * r20;
    r26 = r7 * r3;
    r27 = r5 * r26;
    r28 = r25 + r27;
    r29 = r17 * r28;
    r24 = r24 + r29;
    r24 = fmaf(r14, r24, r0 * r13);
    ReadIdx3<1024, float, float, float4>(
        position, 0 * position_num_alloc, global_thread_idx, r13, r30, r31);
    r22 = fmaf(r1, r22, r2 * r28);
    r24 = fmaf(r13, r14, r24);
    r24 = r24 + r22;
    r13 = 9.99999999999999955e-07;
    r28 = r4 * r4;
    r28 = r28 * r3;
    r12 = r28 + r12;
    r32 = r5 * r6;
    r32 = r32 * r19;
    r33 = r4 * r26;
    r34 = r32 + r33;
    r35 = fmaf(r16, r34, r17 * r12);
    r36 = r5 * r7;
    r36 = r36 * r19;
    r25 = r36 + r25;
    r37 = r15 * r25;
    r35 = r35 + r37;
    r35 = fmaf(r14, r35, r2 * r12);
    r25 = fmaf(r1, r34, r0 * r25);
    r35 = fmaf(r31, r14, r35);
    r35 = r35 + r25;
    r13 = fmaf(r35, r35, r13);
    r9 = r10 + r9;
    r9 = r9 + r28;
    r28 = r7 * r20;
    r32 = r32 + r28;
    r31 = fmaf(r17, r32, r16 * r9);
    r26 = r6 * r26;
    r21 = r21 + r26;
    r31 = fmaf(r15, r21, r31);
    r31 = fmaf(r14, r31, r1 * r9);
    r9 = fmaf(r2, r32, r0 * r21);
    r31 = fmaf(r30, r14, r31);
    r31 = r31 + r9;
    r13 = fmaf(r31, r31, r13);
    r13 = fmaf(r24, r24, r13);
    r30 = rsqrtf(r13);
  };
  LoadUnique<2, float, float>(robust, 0, (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<float>((float*)inout_shared, 0, r12, r38);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r39 = 0.00000000000000000e+00;
    r40 = sqrtf(r13);
    r38 = fmaf(r38, r14, r40);
    r39 = fmaxf(r39, r38);
    r40 = fmaf(r39, r39, r12 * r12);
    r41 = rsqrtf(r40);
    ReadIdx1<1024, float, float, float>(
        sqrt_weight, 0 * sqrt_weight_num_alloc, global_thread_idx, r42);
    r12 = r42 * r12;
    r42 = r39 * r12;
    r43 = r41 * r42;
    r30 = r30 * r43;
    r44 = r24 * r30;
    r45 = r31 * r30;
    r46 = r35 * r30;
    WriteIdx3<1024, float, float, float4>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r44, r45, r46);
    r46 = r19 * r31;
    r45 = r14 * r11;
    r44 = r8 + r45;
    r47 = r7 * r7;
    r48 = r4 * r4;
    r48 = r48 * r14;
    r49 = r47 + r48;
    r50 = r44 + r49;
    r51 = r17 * r14;
    r51 = fmaf(r50, r51, r2 * r50);
    r50 = r16 * r14;
    r51 = fmaf(r34, r50, r51);
    r51 = fmaf(r14, r37, r51);
    r51 = r51 + r25;
    r25 = r19 * r35;
    r37 = r4 * r4;
    r7 = r7 * r7;
    r7 = r7 * r14;
    r50 = r37 + r7;
    r44 = r44 + r50;
    r34 = r4 * r5;
    r34 = r34 * r3;
    r18 = r18 + r34;
    r52 = fmaf(r0, r18, r1 * r44);
    r5 = r5 * r6;
    r5 = r5 * r3;
    r33 = r5 + r33;
    r53 = r16 * r14;
    r52 = fmaf(r44, r53, r52);
    r44 = r17 * r14;
    r52 = fmaf(r33, r44, r52);
    r54 = r15 * r14;
    r52 = fmaf(r18, r54, r52);
    r52 = fmaf(r2, r33, r52);
    r25 = fmaf(r52, r25, r51 * r46);
    r46 = r35 * r25;
    r54 = 2.50000000000000000e-01;
    r41 = r54 * r41;
    r38 = copysign(1.0, r38);
    r38 = r10 + r38;
    r10 = 1.0 / r13;
    r10 = r38 * r10;
    r41 = r41 * r12;
    r41 = r41 * r10;
    r52 = fmaf(r52, r30, r41 * r46);
    r46 = -2.50000000000000000e-01;
    r46 = r39 * r46;
    r39 = r40 * r40;
    r39 = r40 * r39;
    r39 = rsqrtf(r39);
    r46 = r46 * r39;
    r46 = r46 * r42;
    r46 = r46 * r10;
    r10 = r35 * r46;
    r42 = r35 * r25;
    r39 = -5.00000000000000000e-01;
    r40 = r13 * r13;
    r40 = r13 * r40;
    r40 = rsqrtf(r40);
    r40 = r39 * r40;
    r40 = r40 * r43;
    r52 = fmaf(r40, r42, r52);
    r52 = fmaf(r25, r10, r52);
    r42 = r35 * r52;
    r43 = r14 * r30;
    r39 = r31 * r25;
    r13 = r31 * r25;
    r13 = fmaf(r46, r13, r40 * r39);
    r39 = r31 * r41;
    r13 = fmaf(r51, r30, r13);
    r13 = fmaf(r25, r39, r13);
    r51 = r31 * r13;
    r51 = fmaf(r43, r51, r43 * r42);
    r42 = r24 * r40;
    r12 = r24 * r25;
    r12 = fmaf(r41, r12, r25 * r42);
    r38 = r24 * r25;
    r12 = fmaf(r46, r38, r12);
    r38 = r24 * r12;
    r51 = fmaf(r43, r38, r51);
    r38 = r19 * r35;
    r47 = r37 + r47;
    r37 = r14 * r8;
    r47 = r47 + r45;
    r47 = r47 + r37;
    r45 = r15 * r14;
    r45 = fmaf(r47, r45, r0 * r47);
    r45 = fmaf(r14, r23, r45);
    r45 = fmaf(r14, r29, r45);
    r45 = r45 + r22;
    r22 = r19 * r24;
    r37 = r11 + r37;
    r50 = r50 + r37;
    r6 = r4 * r6;
    r6 = r6 * r3;
    r27 = r6 + r27;
    r3 = fmaf(r0, r27, r2 * r50);
    r28 = r5 + r28;
    r5 = r17 * r14;
    r3 = fmaf(r50, r5, r3);
    r50 = r16 * r14;
    r3 = fmaf(r28, r50, r3);
    r29 = r15 * r14;
    r3 = fmaf(r27, r29, r3);
    r3 = fmaf(r1, r28, r3);
    r22 = fmaf(r3, r22, r45 * r38);
    r38 = r35 * r22;
    r45 = fmaf(r45, r30, r41 * r38);
    r38 = r35 * r22;
    r45 = fmaf(r40, r38, r45);
    r45 = fmaf(r22, r10, r45);
    r38 = r35 * r45;
    r29 = r31 * r22;
    r29 = fmaf(r22, r39, r46 * r29);
    r50 = r31 * r22;
    r29 = fmaf(r40, r50, r29);
    r50 = r31 * r29;
    r50 = fmaf(r43, r50, r43 * r38);
    r3 = fmaf(r3, r30, r22 * r42);
    r38 = r24 * r22;
    r3 = fmaf(r46, r38, r3);
    r5 = r24 * r22;
    r3 = fmaf(r41, r5, r3);
    r5 = r24 * r3;
    r50 = fmaf(r43, r5, r50);
    r5 = r11 + r8;
    r5 = r5 + r48;
    r5 = r5 + r7;
    r6 = r36 + r6;
    r2 = fmaf(r2, r6, r0 * r5);
    r26 = r34 + r26;
    r34 = r15 * r14;
    r2 = fmaf(r5, r34, r2);
    r5 = r16 * r14;
    r2 = fmaf(r26, r5, r2);
    r0 = r17 * r14;
    r2 = fmaf(r6, r0, r2);
    r2 = fmaf(r1, r26, r2);
    r0 = r19 * r24;
    r37 = r49 + r37;
    r49 = r16 * r14;
    r49 = fmaf(r37, r49, r1 * r37);
    r37 = r17 * r14;
    r49 = fmaf(r32, r37, r49);
    r32 = r15 * r14;
    r49 = fmaf(r21, r32, r49);
    r49 = r49 + r9;
    r9 = r19 * r31;
    r9 = fmaf(r2, r9, r49 * r0);
    r0 = r31 * r9;
    r0 = fmaf(r46, r0, r2 * r30);
    r2 = r31 * r9;
    r0 = fmaf(r40, r2, r0);
    r0 = fmaf(r9, r39, r0);
    r2 = r31 * r0;
    r32 = r35 * r9;
    r32 = fmaf(r41, r32, r9 * r10);
    r37 = r35 * r9;
    r32 = fmaf(r40, r37, r32);
    r37 = r35 * r32;
    r37 = fmaf(r43, r37, r43 * r2);
    r49 = fmaf(r49, r30, r9 * r42);
    r2 = r24 * r9;
    r49 = fmaf(r41, r2, r49);
    r21 = r24 * r9;
    r49 = fmaf(r46, r21, r49);
    r21 = r24 * r49;
    r37 = fmaf(r43, r21, r37);
    r21 = r19 * r35;
    r2 = r19 * r31;
    r2 = fmaf(r18, r2, r27 * r21);
    r21 = r19 * r24;
    r8 = r19 * r8;
    r11 = fmaf(r19, r11, r14);
    r1 = r8 + r11;
    r2 = fmaf(r1, r21, r2);
    r27 = fmaf(r2, r10, r27 * r30);
    r21 = r35 * r2;
    r27 = fmaf(r41, r21, r27);
    r5 = r35 * r2;
    r27 = fmaf(r40, r5, r27);
    r5 = r35 * r27;
    r1 = fmaf(r1, r30, r2 * r42);
    r21 = r24 * r2;
    r1 = fmaf(r41, r21, r1);
    r34 = r24 * r2;
    r1 = fmaf(r46, r34, r1);
    r34 = r24 * r1;
    r34 = fmaf(r43, r34, r43 * r5);
    r5 = r31 * r2;
    r5 = fmaf(r46, r5, r2 * r39);
    r21 = r31 * r2;
    r5 = fmaf(r40, r21, r5);
    r5 = fmaf(r18, r30, r5);
    r18 = r31 * r5;
    r34 = fmaf(r43, r18, r34);
    WriteSum4<float, float>((float*)inout_shared, r51, r50, r37, r34);
  };
  FlushSumShared<4, float>(out_pose_njtr,
                           0 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r34 = r19 * r35;
    r37 = r19 * r31;
    r8 = r14 + r8;
    r20 = r4 * r20;
    r8 = r8 + r20;
    r37 = fmaf(r8, r37, r28 * r34);
    r34 = r19 * r24;
    r37 = fmaf(r26, r34, r37);
    r26 = fmaf(r26, r30, r37 * r42);
    r34 = r24 * r37;
    r26 = fmaf(r41, r34, r26);
    r4 = r24 * r37;
    r26 = fmaf(r46, r4, r26);
    r4 = r24 * r26;
    r34 = r35 * r37;
    r34 = fmaf(r41, r34, r37 * r10);
    r50 = r35 * r37;
    r34 = fmaf(r40, r50, r34);
    r34 = fmaf(r28, r30, r34);
    r28 = r35 * r34;
    r28 = fmaf(r43, r28, r43 * r4);
    r8 = fmaf(r8, r30, r37 * r39);
    r4 = r31 * r37;
    r8 = fmaf(r40, r4, r8);
    r50 = r31 * r37;
    r8 = fmaf(r46, r50, r8);
    r50 = r31 * r8;
    r28 = fmaf(r43, r50, r28);
    r50 = r19 * r35;
    r11 = r20 + r11;
    r20 = r19 * r31;
    r20 = fmaf(r33, r20, r11 * r50);
    r50 = r19 * r24;
    r20 = fmaf(r6, r50, r20);
    r50 = r24 * r20;
    r6 = fmaf(r6, r30, r41 * r50);
    r50 = r24 * r20;
    r6 = fmaf(r46, r50, r6);
    r6 = fmaf(r20, r42, r6);
    r42 = r24 * r6;
    r50 = r35 * r20;
    r50 = fmaf(r41, r50, r20 * r10);
    r10 = r35 * r20;
    r50 = fmaf(r40, r10, r50);
    r50 = fmaf(r11, r30, r50);
    r11 = r35 * r50;
    r11 = fmaf(r43, r11, r43 * r42);
    r42 = r31 * r20;
    r30 = fmaf(r33, r30, r40 * r42);
    r33 = r31 * r20;
    r30 = fmaf(r46, r33, r30);
    r30 = fmaf(r20, r39, r30);
    r33 = r31 * r30;
    r11 = fmaf(r43, r33, r11);
    WriteSum2<float, float>((float*)inout_shared, r28, r11);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r12, r12, r13 * r13);
    r11 = fmaf(r52, r52, r11);
    r28 = fmaf(r3, r3, r45 * r45);
    r28 = fmaf(r29, r29, r28);
    r33 = fmaf(r0, r0, r49 * r49);
    r33 = fmaf(r32, r32, r33);
    r43 = fmaf(r27, r27, r5 * r5);
    r43 = fmaf(r1, r1, r43);
    WriteSum4<float, float>((float*)inout_shared, r11, r28, r33, r43);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r43 = fmaf(r26, r26, r34 * r34);
    r43 = fmaf(r8, r8, r43);
    r33 = fmaf(r30, r30, r6 * r6);
    r33 = fmaf(r50, r50, r33);
    WriteSum2<float, float>((float*)inout_shared, r43, r33);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fmaf(r13, r29, r12 * r3);
    r33 = fmaf(r52, r45, r33);
    r43 = fmaf(r12, r49, r13 * r0);
    r43 = fmaf(r52, r32, r43);
    r28 = fmaf(r13, r5, r12 * r1);
    r28 = fmaf(r52, r27, r28);
    r11 = fmaf(r13, r8, r52 * r34);
    r11 = fmaf(r12, r26, r11);
    WriteSum4<float, float>((float*)inout_shared, r33, r43, r28, r11);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r52, r50, r13 * r30);
    r11 = fmaf(r12, r6, r11);
    r28 = fmaf(r3, r49, r45 * r32);
    r28 = fmaf(r29, r0, r28);
    r43 = fmaf(r29, r5, r45 * r27);
    r43 = fmaf(r3, r1, r43);
    r33 = fmaf(r3, r26, r45 * r34);
    r33 = fmaf(r29, r8, r33);
    WriteSum4<float, float>((float*)inout_shared, r11, r28, r43, r33);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r33 = fmaf(r29, r30, r3 * r6);
    r33 = fmaf(r45, r50, r33);
    r43 = fmaf(r32, r27, r0 * r5);
    r43 = fmaf(r49, r1, r43);
    r28 = fmaf(r49, r26, r0 * r8);
    r28 = fmaf(r32, r34, r28);
    r11 = fmaf(r49, r6, r32 * r50);
    r11 = fmaf(r0, r30, r11);
    WriteSum4<float, float>((float*)inout_shared, r33, r43, r28, r11);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r27, r34, r5 * r8);
    r11 = fmaf(r1, r26, r11);
    r28 = fmaf(r5, r30, r27 * r50);
    r28 = fmaf(r1, r6, r28);
    r43 = fmaf(r26, r6, r8 * r30);
    r43 = fmaf(r34, r50, r43);
    WriteSum3<float, float>((float*)inout_shared, r11, r28, r43);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
}

void RigSurfaceResJac(float* pose,
                      unsigned int pose_num_alloc,
                      SharedIndex* pose_indices,
                      float* rig_point,
                      unsigned int rig_point_num_alloc,
                      float* position,
                      unsigned int position_num_alloc,
                      float* sqrt_weight,
                      unsigned int sqrt_weight_num_alloc,
                      const float* const robust,
                      float* out_res,
                      unsigned int out_res_num_alloc,
                      float* const out_pose_njtr,
                      unsigned int out_pose_njtr_num_alloc,
                      float* const out_pose_precond_diag,
                      unsigned int out_pose_precond_diag_num_alloc,
                      float* const out_pose_precond_tril,
                      unsigned int out_pose_precond_tril_num_alloc,
                      size_t problem_size) {
  if (problem_size == 0) {
    return;
  }

  const int n_blocks = (problem_size + 1024 - 1) / 1024;
  RigSurfaceResJacKernel<<<n_blocks, 1024>>>(pose,
                                             pose_num_alloc,
                                             pose_indices,
                                             rig_point,
                                             rig_point_num_alloc,
                                             position,
                                             position_num_alloc,
                                             sqrt_weight,
                                             sqrt_weight_num_alloc,
                                             robust,
                                             out_res,
                                             out_res_num_alloc,
                                             out_pose_njtr,
                                             out_pose_njtr_num_alloc,
                                             out_pose_precond_diag,
                                             out_pose_precond_diag_num_alloc,
                                             out_pose_precond_tril,
                                             out_pose_precond_tril_num_alloc,
                                             problem_size);
}

}  // namespace caspar