#include "colmap/sensor/source_coordinate_restoration.h"

#include "colmap/sensor/models.h"
#include "colmap/util/logging.h"

#include <algorithm>

#include <Eigen/Geometry>

namespace colmap {

SourceCoordinateCamera::SourceCoordinateCamera(
    const Eigen::Vector4d& prepared_intrinsics,
    const Eigen::Vector4d& source_intrinsics,
    const std::vector<double>& distortion)
    : prepared_intrinsics_(prepared_intrinsics),
      source_intrinsics_(source_intrinsics) {
  THROW_CHECK(distortion.size() == 0 || distortion.size() == 4 ||
              distortion.size() == 5 || distortion.size() == 8 ||
              distortion.size() == 14);
  std::copy(distortion.begin(), distortion.end(), distortion_.begin());
  // OpenCV tilted-sensor projection: Pz * Ry(-tau_y) * Rx(-tau_x).
  // Model equation: modules/calib3d/src/distortion_model.hpp in opencv/opencv.
  const Eigen::Matrix3d rotation =
      (Eigen::AngleAxisd(-distortion_[13], Eigen::Vector3d::UnitY()) *
       Eigen::AngleAxisd(-distortion_[12], Eigen::Vector3d::UnitX()))
          .toRotationMatrix();
  Eigen::Matrix3d projection = Eigen::Matrix3d::Identity();
  projection(0, 0) = projection(1, 1) = rotation(2, 2);
  projection(0, 2) = -rotation(0, 2);
  projection(1, 2) = -rotation(1, 2);
  tilt_ = projection * rotation;
}

Eigen::Vector2f SourceCoordinateCamera::Restore(
    const Eigen::Vector2f& pixel) const {
  const double x =
      (pixel.x() - prepared_intrinsics_[2]) / prepared_intrinsics_[0];
  const double y =
      (pixel.y() - prepared_intrinsics_[3]) / prepared_intrinsics_[1];
  double dx, dy;
  FullOpenCVCameraModel::Distortion(distortion_.data(), x, y, &dx, &dy);
  const double radius2 = x * x + y * y;
  const double radius4 = radius2 * radius2;
  const Eigen::Vector3d tilted =
      tilt_ *
      Eigen::Vector3d(
          x + dx + distortion_[8] * radius2 + distortion_[9] * radius4,
          y + dy + distortion_[10] * radius2 + distortion_[11] * radius4,
          1.0);
  return Eigen::Vector2f(
      source_intrinsics_[0] * tilted.x() / tilted.z() + source_intrinsics_[2],
      source_intrinsics_[1] * tilted.y() / tilted.z() + source_intrinsics_[3]);
}

}  // namespace colmap
