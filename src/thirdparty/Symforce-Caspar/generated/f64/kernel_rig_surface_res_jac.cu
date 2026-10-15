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
    RigSurfaceResJacKernel(double* pose,
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
                           double* const out_pose_njtr,
                           unsigned int out_pose_njtr_num_alloc,
                           double* const out_pose_precond_diag,
                           unsigned int out_pose_precond_diag_num_alloc,
                           double* const out_pose_precond_tril,
                           unsigned int out_pose_precond_tril_num_alloc,
                           size_t problem_size) {
  const int global_thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ uint8_t inout_shared[16384];

  __shared__ SharedIndex pose_indices_loc[1024];
  pose_indices_loc[threadIdx.x] =
      (global_thread_idx < problem_size
           ? pose_indices[global_thread_idx]
           : SharedIndex{0xffffffff, 0xffff, 0xffff});

  double r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15,
      r16, r17, r18, r19, r20, r21, r22, r23, r24, r25, r26, r27, r28, r29, r30,
      r31, r32, r33, r34, r35, r36, r37, r38, r39, r40, r41, r42, r43, r44, r45,
      r46, r47, r48, r49, r50, r51, r52, r53, r54;

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
    r6 = r2 * r5;
    r7 = 1.00000000000000000e+00;
  };
  LoadShared<2, double, double>(
      pose, 0 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r8, r9);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r10 = r9 * r9;
    r11 = fma(r2, r10, r7);
    r12 = r6 + r11;
    r13 = -1.00000000000000000e+00;
  };
  LoadShared<2, double, double>(
      pose, 4 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r14, r15);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r16 = r3 * r4;
    r17 = 2.00000000000000000e+00;
    r16 = r16 * r17;
    r18 = r8 * r17;
    r19 = r9 * r18;
    r20 = r16 + r19;
    r21 = r15 * r20;
    r22 = fma(r14, r12, r21);
  };
  LoadShared<1, double, double>(
      pose, 6 * pose_num_alloc, pose_indices_loc, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared1<double>(
        (double*)inout_shared, pose_indices_loc[threadIdx.x].target, r23);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r24 = r3 * r18;
    r25 = r4 * r2;
    r26 = r9 * r25;
    r27 = r24 + r26;
    r28 = r23 * r27;
    r22 = r22 + r28;
    r22 = fma(r13, r22, r0 * r12);
    ReadIdx2<1024, double, double, double2>(
        position, 0 * position_num_alloc, global_thread_idx, r12, r29);
    ReadIdx1<1024, double, double, double>(
        rig_point, 2 * rig_point_num_alloc, global_thread_idx, r30);
    r20 = fma(r1, r20, r30 * r27);
    r22 = fma(r12, r13, r22);
    r22 = r22 + r20;
    r12 = 1.00000000000000008e-15;
    r27 = r8 * r8;
    r27 = r27 * r2;
    r11 = r27 + r11;
    r31 = r9 * r3;
    r31 = r31 * r17;
    r32 = r8 * r25;
    r33 = r31 + r32;
    r34 = fma(r15, r33, r23 * r11);
    r35 = r9 * r4;
    r35 = r35 * r17;
    r24 = r35 + r24;
    r36 = r14 * r24;
    r34 = r34 + r36;
    r34 = fma(r13, r34, r30 * r11);
    ReadIdx1<1024, double, double, double>(
        position, 2 * position_num_alloc, global_thread_idx, r11);
    r24 = fma(r1, r33, r0 * r24);
    r34 = fma(r11, r13, r34);
    r34 = r34 + r24;
    r12 = fma(r34, r34, r12);
    r6 = r7 + r6;
    r6 = r6 + r27;
    r27 = r4 * r18;
    r31 = r31 + r27;
    r11 = fma(r23, r31, r15 * r6);
    r25 = r3 * r25;
    r19 = r19 + r25;
    r11 = fma(r14, r19, r11);
    r11 = fma(r13, r11, r1 * r6);
    r6 = fma(r30, r31, r0 * r19);
    r11 = fma(r29, r13, r11);
    r11 = r11 + r6;
    r12 = fma(r11, r11, r12);
    r12 = fma(r22, r22, r12);
    r29 = rsqrt(r12);
  };
  LoadUnique<2, double, double>(robust, 0, (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    ReadShared2<double>((double*)inout_shared, 0, r37, r38);
  };
  __syncthreads();
  if (global_thread_idx < problem_size) {
    r39 = 0.00000000000000000e+00;
    r40 = sqrt(r12);
    r38 = fma(r38, r13, r40);
    r39 = fmax(r39, r38);
    r40 = fma(r39, r39, r37 * r37);
    r41 = rsqrt(r40);
    ReadIdx1<1024, double, double, double>(
        sqrt_weight, 0 * sqrt_weight_num_alloc, global_thread_idx, r42);
    r37 = r42 * r37;
    r42 = r39 * r37;
    r43 = r41 * r42;
    r29 = r29 * r43;
    r44 = r22 * r29;
    r45 = r11 * r29;
    WriteIdx2<1024, double, double, double2>(
        out_res, 0 * out_res_num_alloc, global_thread_idx, r44, r45);
    r45 = r34 * r29;
    WriteIdx1<1024, double, double, double>(
        out_res, 2 * out_res_num_alloc, global_thread_idx, r45);
    r45 = r4 * r4;
    r45 = r45 * r13;
    r44 = r5 + r45;
    r46 = r8 * r8;
    r47 = r13 * r10;
    r48 = r46 + r47;
    r49 = r44 + r48;
    r50 = r8 * r9;
    r50 = r50 * r2;
    r16 = r16 + r50;
    r51 = fma(r0, r16, r1 * r49);
    r9 = r9 * r3;
    r9 = r9 * r2;
    r32 = r9 + r32;
    r52 = r15 * r13;
    r51 = fma(r49, r52, r51);
    r49 = r23 * r13;
    r51 = fma(r32, r49, r51);
    r53 = r14 * r13;
    r51 = fma(r16, r53, r51);
    r51 = fma(r30, r32, r51);
    r53 = r17 * r11;
    r47 = r5 + r47;
    r4 = r4 * r4;
    r49 = r8 * r8;
    r49 = r49 * r13;
    r52 = r4 + r49;
    r47 = r47 + r52;
    r54 = r23 * r13;
    r54 = fma(r47, r54, r30 * r47);
    r47 = r15 * r13;
    r54 = fma(r33, r47, r54);
    r54 = fma(r13, r36, r54);
    r54 = r54 + r24;
    r24 = r17 * r34;
    r24 = fma(r51, r24, r54 * r53);
    r53 = 2.50000000000000000e-01;
    r41 = r53 * r41;
    r38 = copysign(1.0, r38);
    r38 = r7 + r38;
    r7 = 1.0 / r12;
    r7 = r38 * r7;
    r41 = r41 * r37;
    r41 = r41 * r7;
    r37 = r34 * r41;
    r51 = fma(r24, r37, r51 * r29);
    r38 = r34 * r24;
    r53 = -2.50000000000000000e-01;
    r53 = r39 * r53;
    r39 = r40 * r40;
    r39 = r40 * r39;
    r39 = rsqrt(r39);
    r53 = r53 * r39;
    r53 = r53 * r42;
    r53 = r53 * r7;
    r51 = fma(r53, r38, r51);
    r7 = r34 * r24;
    r42 = -5.00000000000000000e-01;
    r39 = r12 * r12;
    r39 = r12 * r39;
    r39 = rsqrt(r39);
    r39 = r42 * r39;
    r39 = r39 * r43;
    r51 = fma(r39, r7, r51);
    r7 = r34 * r51;
    r38 = r13 * r29;
    r43 = r22 * r53;
    r42 = r22 * r24;
    r42 = fma(r41, r42, r24 * r43);
    r12 = r22 * r24;
    r42 = fma(r39, r12, r42);
    r12 = r22 * r42;
    r12 = fma(r38, r12, r38 * r7);
    r7 = r11 * r24;
    r54 = fma(r54, r29, r53 * r7);
    r7 = r11 * r39;
    r40 = r11 * r24;
    r54 = fma(r41, r40, r54);
    r54 = fma(r24, r7, r54);
    r40 = r11 * r54;
    r12 = fma(r38, r40, r12);
    r40 = r17 * r34;
    r36 = r13 * r5;
    r4 = r4 + r36;
    r4 = r4 + r48;
    r48 = r14 * r13;
    r48 = fma(r4, r48, r0 * r4);
    r48 = fma(r13, r21, r48);
    r48 = fma(r13, r28, r48);
    r48 = r48 + r20;
    r20 = r17 * r22;
    r45 = r46 + r45;
    r36 = r10 + r36;
    r45 = r45 + r36;
    r3 = r8 * r3;
    r3 = r3 * r2;
    r26 = r3 + r26;
    r2 = fma(r0, r26, r30 * r45);
    r27 = r9 + r27;
    r9 = r23 * r13;
    r2 = fma(r45, r9, r2);
    r45 = r15 * r13;
    r2 = fma(r27, r45, r2);
    r46 = r14 * r13;
    r2 = fma(r26, r46, r2);
    r2 = fma(r1, r27, r2);
    r20 = fma(r2, r20, r48 * r40);
    r40 = r11 * r20;
    r40 = fma(r20, r7, r53 * r40);
    r46 = r11 * r20;
    r40 = fma(r41, r46, r40);
    r46 = r11 * r40;
    r45 = r22 * r20;
    r2 = fma(r2, r29, r39 * r45);
    r45 = r22 * r20;
    r2 = fma(r41, r45, r2);
    r2 = fma(r20, r43, r2);
    r45 = r22 * r2;
    r45 = fma(r38, r45, r38 * r46);
    r46 = r34 * r20;
    r46 = fma(r39, r46, r48 * r29);
    r48 = r34 * r20;
    r46 = fma(r53, r48, r46);
    r46 = fma(r20, r37, r46);
    r48 = r34 * r46;
    r45 = fma(r38, r48, r45);
    WriteSum2<double, double>((double*)inout_shared, r12, r45);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            0 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r45 = r17 * r22;
    r36 = r52 + r36;
    r52 = r15 * r13;
    r52 = fma(r36, r52, r1 * r36);
    r36 = r23 * r13;
    r52 = fma(r31, r36, r52);
    r31 = r14 * r13;
    r52 = fma(r19, r31, r52);
    r52 = r52 + r6;
    r6 = r17 * r11;
    r49 = r10 + r49;
    r49 = r49 + r44;
    r3 = r35 + r3;
    r30 = fma(r30, r3, r0 * r49);
    r25 = r50 + r25;
    r50 = r14 * r13;
    r30 = fma(r49, r50, r30);
    r49 = r15 * r13;
    r30 = fma(r25, r49, r30);
    r0 = r23 * r13;
    r30 = fma(r3, r0, r30);
    r30 = fma(r1, r25, r30);
    r6 = fma(r30, r6, r52 * r45);
    r45 = r34 * r6;
    r45 = fma(r6, r37, r53 * r45);
    r0 = r34 * r6;
    r45 = fma(r39, r0, r45);
    r0 = r34 * r45;
    r49 = r22 * r6;
    r49 = fma(r41, r49, r52 * r29);
    r52 = r22 * r6;
    r49 = fma(r39, r52, r49);
    r49 = fma(r6, r43, r49);
    r52 = r22 * r49;
    r52 = fma(r38, r52, r38 * r0);
    r0 = r11 * r6;
    r0 = fma(r41, r0, r30 * r29);
    r30 = r11 * r6;
    r0 = fma(r53, r30, r0);
    r0 = fma(r6, r7, r0);
    r30 = r11 * r0;
    r52 = fma(r38, r30, r52);
    r30 = r17 * r34;
    r50 = r17 * r11;
    r50 = fma(r16, r50, r26 * r30);
    r30 = r17 * r22;
    r5 = r17 * r5;
    r10 = fma(r17, r10, r13);
    r1 = r5 + r10;
    r50 = fma(r1, r30, r50);
    r30 = r34 * r50;
    r26 = fma(r26, r29, r39 * r30);
    r30 = r34 * r50;
    r26 = fma(r53, r30, r26);
    r26 = fma(r50, r37, r26);
    r30 = r34 * r26;
    r16 = fma(r50, r7, r16 * r29);
    r35 = r11 * r50;
    r16 = fma(r41, r35, r16);
    r44 = r11 * r50;
    r16 = fma(r53, r44, r16);
    r44 = r11 * r16;
    r44 = fma(r38, r44, r38 * r30);
    r30 = r22 * r50;
    r30 = fma(r50, r43, r39 * r30);
    r35 = r22 * r50;
    r30 = fma(r41, r35, r30);
    r30 = fma(r1, r29, r30);
    r35 = r22 * r30;
    r44 = fma(r38, r35, r44);
    WriteSum2<double, double>((double*)inout_shared, r52, r44);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            2 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r44 = r17 * r34;
    r52 = r17 * r11;
    r5 = r13 + r5;
    r18 = r8 * r18;
    r5 = r5 + r18;
    r52 = fma(r5, r52, r27 * r44);
    r44 = r17 * r22;
    r52 = fma(r25, r44, r52);
    r44 = r22 * r52;
    r44 = fma(r41, r44, r52 * r43);
    r8 = r22 * r52;
    r44 = fma(r39, r8, r44);
    r44 = fma(r25, r29, r44);
    r8 = r22 * r44;
    r25 = r11 * r52;
    r35 = r11 * r52;
    r35 = fma(r41, r35, r53 * r25);
    r35 = fma(r52, r7, r35);
    r35 = fma(r5, r29, r35);
    r5 = r11 * r35;
    r5 = fma(r38, r5, r38 * r8);
    r8 = r34 * r52;
    r8 = fma(r52, r37, r39 * r8);
    r25 = r34 * r52;
    r8 = fma(r53, r25, r8);
    r8 = fma(r27, r29, r8);
    r27 = r34 * r8;
    r5 = fma(r38, r27, r5);
    r27 = r17 * r34;
    r10 = r18 + r10;
    r18 = r17 * r11;
    r18 = fma(r32, r18, r10 * r27);
    r27 = r17 * r22;
    r18 = fma(r3, r27, r18);
    r27 = r34 * r18;
    r10 = fma(r10, r29, r53 * r27);
    r27 = r34 * r18;
    r10 = fma(r39, r27, r10);
    r10 = fma(r18, r37, r10);
    r27 = r34 * r10;
    r37 = r11 * r18;
    r25 = r11 * r18;
    r25 = fma(r41, r25, r53 * r37);
    r25 = fma(r18, r7, r25);
    r25 = fma(r32, r29, r25);
    r32 = r11 * r25;
    r32 = fma(r38, r32, r38 * r27);
    r27 = r22 * r18;
    r7 = r22 * r18;
    r7 = fma(r39, r7, r41 * r27);
    r7 = fma(r3, r29, r7);
    r7 = fma(r18, r43, r7);
    r43 = r22 * r7;
    r32 = fma(r38, r43, r32);
    WriteSum2<double, double>((double*)inout_shared, r5, r32);
  };
  FlushSumShared<2, double>(out_pose_njtr,
                            4 * out_pose_njtr_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r54, r54, r51 * r51);
    r32 = fma(r42, r42, r32);
    r5 = fma(r40, r40, r2 * r2);
    r5 = fma(r46, r46, r5);
    WriteSum2<double, double>((double*)inout_shared, r32, r5);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            0 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r5 = fma(r49, r49, r45 * r45);
    r5 = fma(r0, r0, r5);
    r32 = fma(r30, r30, r16 * r16);
    r32 = fma(r26, r26, r32);
    WriteSum2<double, double>((double*)inout_shared, r5, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            2 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r8, r8, r44 * r44);
    r32 = fma(r35, r35, r32);
    r5 = fma(r25, r25, r10 * r10);
    r5 = fma(r7, r7, r5);
    WriteSum2<double, double>((double*)inout_shared, r32, r5);
  };
  FlushSumShared<2, double>(out_pose_precond_diag,
                            4 * out_pose_precond_diag_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r5 = fma(r51, r46, r54 * r40);
    r5 = fma(r42, r2, r5);
    r32 = fma(r51, r45, r54 * r0);
    r32 = fma(r42, r49, r32);
    WriteSum2<double, double>((double*)inout_shared, r5, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            0 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r54, r16, r51 * r26);
    r32 = fma(r42, r30, r32);
    r5 = fma(r51, r8, r54 * r35);
    r5 = fma(r42, r44, r5);
    WriteSum2<double, double>((double*)inout_shared, r32, r5);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            2 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r5 = fma(r42, r7, r54 * r25);
    r5 = fma(r51, r10, r5);
    r32 = fma(r2, r49, r40 * r0);
    r32 = fma(r46, r45, r32);
    WriteSum2<double, double>((double*)inout_shared, r5, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            4 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r2, r30, r40 * r16);
    r32 = fma(r46, r26, r32);
    r5 = fma(r40, r35, r46 * r8);
    r5 = fma(r2, r44, r5);
    WriteSum2<double, double>((double*)inout_shared, r32, r5);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            6 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r5 = fma(r2, r7, r46 * r10);
    r5 = fma(r40, r25, r5);
    r32 = fma(r0, r16, r45 * r26);
    r32 = fma(r49, r30, r32);
    WriteSum2<double, double>((double*)inout_shared, r5, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            8 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r0, r35, r49 * r44);
    r32 = fma(r45, r8, r32);
    r5 = fma(r45, r10, r0 * r25);
    r5 = fma(r49, r7, r5);
    WriteSum2<double, double>((double*)inout_shared, r32, r5);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            10 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r5 = fma(r16, r35, r26 * r8);
    r5 = fma(r30, r44, r5);
    r32 = fma(r26, r10, r30 * r7);
    r32 = fma(r16, r25, r32);
    WriteSum2<double, double>((double*)inout_shared, r5, r32);
  };
  FlushSumShared<2, double>(out_pose_precond_tril,
                            12 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
  if (global_thread_idx < problem_size) {
    r32 = fma(r8, r10, r35 * r25);
    r32 = fma(r44, r7, r32);
    WriteSum1<double, double>((double*)inout_shared, r32);
  };
  FlushSumShared<1, double>(out_pose_precond_tril,
                            14 * out_pose_precond_tril_num_alloc,
                            pose_indices_loc,
                            (double*)inout_shared);
}

void RigSurfaceResJac(double* pose,
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
                      double* const out_pose_njtr,
                      unsigned int out_pose_njtr_num_alloc,
                      double* const out_pose_precond_diag,
                      unsigned int out_pose_precond_diag_num_alloc,
                      double* const out_pose_precond_tril,
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