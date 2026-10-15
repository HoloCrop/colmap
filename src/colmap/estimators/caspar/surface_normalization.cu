#include "colmap/estimators/caspar/surface_normalization.h"

#include <stdexcept>

#include <cuda_runtime.h>

namespace colmap {
namespace {

constexpr int kBlockSize = 256;

void CheckLaunch() {
  const cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(status));
  }
}

unsigned int GridSize(const size_t num) {
  return static_cast<unsigned int>((num + kBlockSize - 1) / kBlockSize);
}

__global__ void AffineTransformPointsKernel(const float* source,
                                            float* target,
                                            const size_t num,
                                            const float scale,
                                            const float3 translation) {
  const size_t index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (index >= num) {
    return;
  }
  target[3 * index + 0] = scale * source[3 * index + 0] + translation.x;
  target[3 * index + 1] = scale * source[3 * index + 1] + translation.y;
  target[3 * index + 2] = scale * source[3 * index + 2] + translation.z;
}

__global__ void ScaleKernel(const float* source,
                            float* target,
                            const size_t num,
                            const float scale) {
  const size_t index = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x;
  if (index >= num) {
    return;
  }
  target[index] = scale * source[index];
}

}  // namespace

void AffineTransformPointsDevice(const float* source,
                                 float* target,
                                 const size_t num,
                                 const float scale,
                                 const float translation_x,
                                 const float translation_y,
                                 const float translation_z) {
  if (num == 0) {
    return;
  }
  AffineTransformPointsKernel<<<GridSize(num), kBlockSize>>>(
      source,
      target,
      num,
      scale,
      make_float3(translation_x, translation_y, translation_z));
  CheckLaunch();
}

void ScaleDevice(const float* source,
                 float* target,
                 const size_t num,
                 const float scale) {
  if (num == 0) {
    return;
  }
  ScaleKernel<<<GridSize(num), kBlockSize>>>(source, target, num, scale);
  CheckLaunch();
}

}  // namespace colmap
