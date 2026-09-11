#pragma once

#include <array>
#include <vector>

#include <Eigen/Core>

namespace colmap {

// Alignment export only: map prepared pinhole pixels back to source pixels.
// Supports OpenCV's 0/4/5/8/14 coefficient models without computing
// derivatives.
class SourceCoordinateCamera {
 public:
  SourceCoordinateCamera(const Eigen::Vector4d& prepared_intrinsics,
                         const Eigen::Vector4d& source_intrinsics,
                         const std::vector<double>& distortion);
  Eigen::Vector2f Restore(const Eigen::Vector2f& pixel) const;

 private:
  Eigen::Vector4d prepared_intrinsics_;
  Eigen::Vector4d source_intrinsics_;
  std::array<double, 14> distortion_{};
  Eigen::Matrix3d tilt_;
};

}  // namespace colmap
