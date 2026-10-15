#include "kernel_rig_surface_res_jac_first.h"
#include "memops.cuh"
#include <cooperative_groups.h>
#include <cooperative_groups/details/partitioning.h>
#include <cooperative_groups/memcpy_async.h>
#include <cooperative_groups/reduce.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

namespace caspar {

__global__ void __launch_bounds__(1024, 1)
    RigSurfaceResJacFirstKernel(float* pose,
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
                                float* const out_rTr,
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

  __shared__ float out_rTr_local[1];

  float r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54, r55;

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
    r43 = r42 * r12;
    r44 = r39 * r43;
    r45 = r41 * r44;
    r30 = r30 * r45;
    r46 = r24 * r30;
    r47 = r31 * r30;
    r48 = r35 * r30;
    WriteIdx3<1024, float, float, float4>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r46, r47, r48);
    r48 = r35 * r35;
    r12 = r42 * r12;
    r42 = 1.0 / r40;
    r47 = 1.0 / r13;
    r12 = r12 * r39;
    r12 = r12 * r42;
    r12 = r12 * r47;
    r12 = r12 * r44;
    r42 = r24 * r24;
    r42 = fmaf(r12, r42, r12 * r48);
    r48 = r31 * r31;
    r42 = fmaf(r12, r48, r42);
  };
  SumStore<float>(out_rTr_local,
                  (float*)inout_shared,
                  0,
                  global_thread_idx < problem_size,
                  r42);
  if (global_thread_idx < problem_size) {
    r42 = r19 * r31;
    r48 = r14 * r11;
    r12 = r8 + r48;
    r46 = r7 * r7;
    r49 = r4 * r4;
    r49 = r49 * r14;
    r50 = r46 + r49;
    r51 = r12 + r50;
    r52 = r17 * r14;
    r52 = fmaf(r51, r52, r2 * r51);
    r51 = r16 * r14;
    r52 = fmaf(r34, r51, r52);
    r52 = fmaf(r14, r37, r52);
    r52 = r52 + r25;
    r25 = r19 * r35;
    r37 = r4 * r4;
    r7 = r7 * r7;
    r7 = r7 * r14;
    r51 = r37 + r7;
    r12 = r12 + r51;
    r34 = r4 * r5;
    r34 = r34 * r3;
    r18 = r18 + r34;
    r53 = fmaf(r0, r18, r1 * r12);
    r5 = r5 * r6;
    r5 = r5 * r3;
    r33 = r5 + r33;
    r54 = r16 * r14;
    r53 = fmaf(r12, r54, r53);
    r12 = r17 * r14;
    r53 = fmaf(r33, r12, r53);
    r55 = r15 * r14;
    r53 = fmaf(r18, r55, r53);
    r53 = fmaf(r2, r33, r53);
    r25 = fmaf(r53, r25, r52 * r42);
    r42 = 2.50000000000000000e-01;
    r41 = r42 * r41;
    r38 = copysign(1.0, r38);
    r38 = r10 + r38;
    r47 = r38 * r47;
    r41 = r41 * r43;
    r41 = r41 * r47;
    r43 = r35 * r41;
    r53 = fmaf(r53, r30, r25 * r43);
    r38 = r35 * r25;
    r10 = -2.50000000000000000e-01;
    r10 = r39 * r10;
    r39 = r40 * r40;
    r39 = r40 * r39;
    r39 = rsqrtf(r39);
    r10 = r10 * r39;
    r10 = r10 * r44;
    r10 = r10 * r47;
    r53 = fmaf(r10, r38, r53);
    r47 = r35 * r25;
    r44 = -5.00000000000000000e-01;
    r39 = r13 * r13;
    r39 = r13 * r39;
    r39 = rsqrtf(r39);
    r39 = r44 * r39;
    r39 = r39 * r45;
    r53 = fmaf(r39, r47, r53);
    r47 = r35 * r53;
    r38 = r14 * r30;
    r45 = r31 * r25;
    r44 = r31 * r10;
    r45 = fmaf(r25, r44, r39 * r45);
    r13 = r31 * r25;
    r45 = fmaf(r41, r13, r45);
    r45 = fmaf(r52, r30, r45);
    r13 = r31 * r45;
    r13 = fmaf(r38, r13, r38 * r47);
    r47 = r24 * r39;
    r52 = r24 * r25;
    r52 = fmaf(r41, r52, r25 * r47);
    r40 = r24 * r25;
    r52 = fmaf(r10, r40, r52);
    r40 = r24 * r52;
    r13 = fmaf(r38, r40, r13);
    r40 = r19 * r35;
    r46 = r37 + r46;
    r37 = r14 * r8;
    r46 = r46 + r48;
    r46 = r46 + r37;
    r48 = r15 * r14;
    r48 = fmaf(r46, r48, r0 * r46);
    r48 = fmaf(r14, r23, r48);
    r48 = fmaf(r14, r29, r48);
    r48 = r48 + r22;
    r22 = r19 * r24;
    r37 = r11 + r37;
    r51 = r51 + r37;
    r6 = r4 * r6;
    r6 = r6 * r3;
    r27 = r6 + r27;
    r3 = fmaf(r0, r27, r2 * r51);
    r28 = r5 + r28;
    r5 = r17 * r14;
    r3 = fmaf(r51, r5, r3);
    r51 = r16 * r14;
    r3 = fmaf(r28, r51, r3);
    r29 = r15 * r14;
    r3 = fmaf(r27, r29, r3);
    r3 = fmaf(r1, r28, r3);
    r22 = fmaf(r3, r22, r48 * r40);
    r48 = fmaf(r48, r30, r22 * r43);
    r40 = r35 * r22;
    r48 = fmaf(r10, r40, r48);
    r29 = r35 * r22;
    r48 = fmaf(r39, r29, r48);
    r29 = r35 * r48;
    r40 = r31 * r22;
    r40 = fmaf(r41, r40, r22 * r44);
    r51 = r31 * r22;
    r40 = fmaf(r39, r51, r40);
    r51 = r31 * r40;
    r51 = fmaf(r38, r51, r38 * r29);
    r3 = fmaf(r3, r30, r22 * r47);
    r29 = r24 * r22;
    r3 = fmaf(r10, r29, r3);
    r5 = r24 * r22;
    r3 = fmaf(r41, r5, r3);
    r5 = r24 * r3;
    r51 = fmaf(r38, r5, r51);
    r5 = r11 + r8;
    r5 = r5 + r49;
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
    r37 = r50 + r37;
    r50 = r16 * r14;
    r50 = fmaf(r37, r50, r1 * r37);
    r37 = r17 * r14;
    r50 = fmaf(r32, r37, r50);
    r32 = r15 * r14;
    r50 = fmaf(r21, r32, r50);
    r50 = r50 + r9;
    r9 = r19 * r31;
    r9 = fmaf(r2, r9, r50 * r0);
    r2 = fmaf(r9, r44, r2 * r30);
    r0 = r31 * r9;
    r2 = fmaf(r39, r0, r2);
    r32 = r31 * r9;
    r2 = fmaf(r41, r32, r2);
    r32 = r31 * r2;
    r0 = r35 * r9;
    r0 = fmaf(r9, r43, r10 * r0);
    r37 = r35 * r9;
    r0 = fmaf(r39, r37, r0);
    r37 = r35 * r0;
    r37 = fmaf(r38, r37, r38 * r32);
    r50 = fmaf(r50, r30, r9 * r47);
    r32 = r24 * r9;
    r50 = fmaf(r41, r32, r50);
    r21 = r24 * r9;
    r50 = fmaf(r10, r21, r50);
    r21 = r24 * r50;
    r37 = fmaf(r38, r21, r37);
    r21 = r19 * r35;
    r32 = r19 * r31;
    r32 = fmaf(r18, r32, r27 * r21);
    r21 = r19 * r24;
    r8 = r19 * r8;
    r11 = fmaf(r19, r11, r14);
    r1 = r8 + r11;
    r32 = fmaf(r1, r21, r32);
    r21 = r35 * r32;
    r21 = fmaf(r10, r21, r27 * r30);
    r27 = r35 * r32;
    r21 = fmaf(r39, r27, r21);
    r21 = fmaf(r32, r43, r21);
    r27 = r35 * r21;
    r1 = fmaf(r1, r30, r32 * r47);
    r5 = r24 * r32;
    r1 = fmaf(r41, r5, r1);
    r34 = r24 * r32;
    r1 = fmaf(r10, r34, r1);
    r34 = r24 * r1;
    r34 = fmaf(r38, r34, r38 * r27);
    r27 = r31 * r32;
    r27 = fmaf(r32, r44, r41 * r27);
    r5 = r31 * r32;
    r27 = fmaf(r39, r5, r27);
    r27 = fmaf(r18, r30, r27);
    r18 = r31 * r27;
    r34 = fmaf(r38, r18, r34);
    WriteSum4<float, float>((float*)inout_shared, r13, r51, r37, r34);
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
    r26 = fmaf(r26, r30, r37 * r47);
    r34 = r24 * r37;
    r26 = fmaf(r41, r34, r26);
    r4 = r24 * r37;
    r26 = fmaf(r10, r4, r26);
    r4 = r24 * r26;
    r34 = r35 * r37;
    r34 = fmaf(r37, r43, r10 * r34);
    r51 = r35 * r37;
    r34 = fmaf(r39, r51, r34);
    r34 = fmaf(r28, r30, r34);
    r28 = r35 * r34;
    r28 = fmaf(r38, r28, r38 * r4);
    r4 = r31 * r37;
    r8 = fmaf(r8, r30, r41 * r4);
    r4 = r31 * r37;
    r8 = fmaf(r39, r4, r8);
    r8 = fmaf(r37, r44, r8);
    r4 = r31 * r8;
    r28 = fmaf(r38, r4, r28);
    r4 = r19 * r35;
    r11 = r20 + r11;
    r20 = r19 * r31;
    r20 = fmaf(r33, r20, r11 * r4);
    r4 = r19 * r24;
    r20 = fmaf(r6, r4, r20);
    r4 = r24 * r20;
    r6 = fmaf(r6, r30, r41 * r4);
    r4 = r24 * r20;
    r6 = fmaf(r10, r4, r6);
    r6 = fmaf(r20, r47, r6);
    r47 = r24 * r6;
    r4 = r35 * r20;
    r43 = fmaf(r20, r43, r10 * r4);
    r4 = r35 * r20;
    r43 = fmaf(r39, r4, r43);
    r43 = fmaf(r11, r30, r43);
    r11 = r35 * r43;
    r11 = fmaf(r38, r11, r38 * r47);
    r47 = r31 * r20;
    r30 = fmaf(r33, r30, r39 * r47);
    r33 = r31 * r20;
    r30 = fmaf(r41, r33, r30);
    r30 = fmaf(r20, r44, r30);
    r44 = r31 * r30;
    r11 = fmaf(r38, r44, r11);
    WriteSum2<float, float>((float*)inout_shared, r28, r11);
  };
  FlushSumShared<2, float>(out_pose_njtr,
                           4 * out_pose_njtr_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r52, r52, r45 * r45);
    r11 = fmaf(r53, r53, r11);
    r28 = fmaf(r3, r3, r48 * r48);
    r28 = fmaf(r40, r40, r28);
    r44 = fmaf(r2, r2, r50 * r50);
    r44 = fmaf(r0, r0, r44);
    r38 = fmaf(r21, r21, r27 * r27);
    r38 = fmaf(r1, r1, r38);
    WriteSum4<float, float>((float*)inout_shared, r11, r28, r44, r38);
  };
  FlushSumShared<4, float>(out_pose_precond_diag,
                           0 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r38 = fmaf(r26, r26, r34 * r34);
    r38 = fmaf(r8, r8, r38);
    r44 = fmaf(r30, r30, r6 * r6);
    r44 = fmaf(r43, r43, r44);
    WriteSum2<float, float>((float*)inout_shared, r38, r44);
  };
  FlushSumShared<2, float>(out_pose_precond_diag,
                           4 * out_pose_precond_diag_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fmaf(r45, r40, r52 * r3);
    r44 = fmaf(r53, r48, r44);
    r38 = fmaf(r52, r50, r45 * r2);
    r38 = fmaf(r53, r0, r38);
    r28 = fmaf(r45, r27, r52 * r1);
    r28 = fmaf(r53, r21, r28);
    r11 = fmaf(r45, r8, r53 * r34);
    r11 = fmaf(r52, r26, r11);
    WriteSum4<float, float>((float*)inout_shared, r44, r38, r28, r11);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           0 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r53, r43, r45 * r30);
    r11 = fmaf(r52, r6, r11);
    r28 = fmaf(r3, r50, r48 * r0);
    r28 = fmaf(r40, r2, r28);
    r38 = fmaf(r40, r27, r48 * r21);
    r38 = fmaf(r3, r1, r38);
    r44 = fmaf(r3, r26, r48 * r34);
    r44 = fmaf(r40, r8, r44);
    WriteSum4<float, float>((float*)inout_shared, r11, r28, r38, r44);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           4 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = fmaf(r40, r30, r3 * r6);
    r44 = fmaf(r48, r43, r44);
    r38 = fmaf(r0, r21, r2 * r27);
    r38 = fmaf(r50, r1, r38);
    r28 = fmaf(r50, r26, r2 * r8);
    r28 = fmaf(r0, r34, r28);
    r11 = fmaf(r50, r6, r0 * r43);
    r11 = fmaf(r2, r30, r11);
    WriteSum4<float, float>((float*)inout_shared, r44, r38, r28, r11);
  };
  FlushSumShared<4, float>(out_pose_precond_tril,
                           8 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  if (global_thread_idx < problem_size) {
    r11 = fmaf(r21, r34, r27 * r8);
    r11 = fmaf(r1, r26, r11);
    r28 = fmaf(r27, r30, r21 * r43);
    r28 = fmaf(r1, r6, r28);
    r38 = fmaf(r26, r6, r8 * r30);
    r38 = fmaf(r34, r43, r38);
    WriteSum3<float, float>((float*)inout_shared, r11, r28, r38);
  };
  FlushSumShared<3, float>(out_pose_precond_tril,
                           12 * out_pose_precond_tril_num_alloc,
                           pose_indices_loc,
                           (float*)inout_shared);
  SumFlushFinal<float>(out_rTr_local, out_rTr, 1);
}

void RigSurfaceResJacFirst(float* pose,
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
                           float* const out_rTr,
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
  RigSurfaceResJacFirstKernel<<<n_blocks, 1024>>>(
      pose,
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
      out_rTr,
      out_pose_njtr,
      out_pose_njtr_num_alloc,
      out_pose_precond_diag,
      out_pose_precond_diag_num_alloc,
      out_pose_precond_tril,
      out_pose_precond_tril_num_alloc,
      problem_size);
}

}  // namespace caspar