#pragma once

#include <cstddef>

namespace colmap {

// target = scale * source + translation for `num` row-major 3-vectors in
// device memory. `target` may alias `source`.
void AffineTransformPointsDevice(const float* source,
                                 float* target,
                                 size_t num,
                                 float scale,
                                 float translation_x,
                                 float translation_y,
                                 float translation_z);

// target = scale * source for `num` floats in device memory.
void ScaleDevice(const float* source, float* target, size_t num, float scale);

}  // namespace colmap
