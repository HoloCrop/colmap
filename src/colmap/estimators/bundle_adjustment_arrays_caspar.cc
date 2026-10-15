#include "colmap/estimators/bundle_adjustment_arrays_caspar.h"

#include "colmap/geometry/normalization.h"
#include "colmap/geometry/pose.h"
#include "colmap/util/logging.h"

#include <utility>

#include <Eigen/Core>

namespace colmap::bundle_adjustment_arrays {
namespace {

using PackedPose = Eigen::Matrix<StorageType, 7, 1>;
using PackedSensorCalibration = Eigen::Matrix<StorageType, 11, 1>;
using InputCalibration = Eigen::Matrix<float, 4, 1>;

constexpr double kNormalizationExtent = 10.0;
constexpr double kNormalizationMinPercentile = 0.1;
constexpr double kNormalizationMaxPercentile = 0.9;

template <typename ImageIndex>
Sim3d ComputeNormalizationImpl(const std::vector<Rigid3d>& rigs_from_world,
                               const uint32_t* image_frame_indices,
                               const uint32_t* image_sensor_indices,
                               const size_t num_selected_images,
                               const std::vector<Rigid3d>& sensors_from_rig,
                               ImageIndex image_index) {
  std::vector<double> centers_x;
  std::vector<double> centers_y;
  std::vector<double> centers_z;
  centers_x.reserve(num_selected_images);
  centers_y.reserve(num_selected_images);
  centers_z.reserve(num_selected_images);
  for (size_t index = 0; index < num_selected_images; ++index) {
    const size_t image = image_index(index);
    const Rigid3d sensor_from_world =
        sensors_from_rig[image_sensor_indices[image]] *
        rigs_from_world[image_frame_indices[image]];
    const Eigen::Vector3d center = sensor_from_world.TgtOriginInSrc();
    centers_x.push_back(center.x());
    centers_y.push_back(center.y());
    centers_z.push_back(center.z());
  }
  const auto [bounding_box, centroid] =
      ComputeBoundingBoxAndCentroid(kNormalizationMinPercentile,
                                    kNormalizationMaxPercentile,
                                    std::move(centers_x),
                                    std::move(centers_y),
                                    std::move(centers_z));
  const double extent = bounding_box.diagonal().norm();
  THROW_CHECK_GT(extent, 0.0)
      << "Bundle adjustment requires non-degenerate camera positions";
  const double scale = kNormalizationExtent / extent;
  return Sim3d(scale, Eigen::Quaterniond::Identity(), -scale * centroid);
}

}  // namespace

Sim3d ComputeNormalization(const std::vector<Rigid3d>& rigs_from_world,
                           const uint32_t* image_frame_indices,
                           const uint32_t* image_sensor_indices,
                           const size_t num_images,
                           const std::vector<Rigid3d>& sensors_from_rig) {
  return ComputeNormalizationImpl(rigs_from_world,
                                  image_frame_indices,
                                  image_sensor_indices,
                                  num_images,
                                  sensors_from_rig,
                                  [](const size_t image) { return image; });
}

Sim3d ComputeNormalization(const std::vector<Rigid3d>& rigs_from_world,
                           const uint32_t* image_frame_indices,
                           const uint32_t* image_sensor_indices,
                           const uint32_t* selected_image_indices,
                           const size_t num_selected_images,
                           const std::vector<Rigid3d>& sensors_from_rig) {
  return ComputeNormalizationImpl(rigs_from_world,
                                  image_frame_indices,
                                  image_sensor_indices,
                                  num_selected_images,
                                  sensors_from_rig,
                                  [selected_image_indices](const size_t index) {
                                    return selected_image_indices[index];
                                  });
}

NormalizedProblem NormalizeProblem(
    const Sim3d& normalized_from_metric,
    const std::vector<Rigid3d>& initial_rigs_from_world,
    const float* initial_points,
    const size_t num_points,
    const std::vector<Rigid3d>& sensors_from_rig) {
  NormalizedProblem problem{
      normalized_from_metric,
      initial_rigs_from_world,
      sensors_from_rig,
      std::vector<StorageType>(initial_points, initial_points + 3 * num_points),
  };
  TransformPoses(problem.rigs_from_world, problem.normalized_from_metric);
  for (Rigid3d& sensor_from_rig : problem.sensors_from_rig) {
    sensor_from_rig.translation() *= problem.normalized_from_metric.scale();
  }
  TransformPoints(problem.points, problem.normalized_from_metric);
  return problem;
}

std::vector<StorageType> PackSensorCalibrations(
    const std::vector<Rigid3d>& sensors_from_rig,
    const float* sensor_calibrations) {
  std::vector<StorageType> packed(11 * sensors_from_rig.size());
  for (size_t sensor = 0; sensor < sensors_from_rig.size(); ++sensor) {
    Eigen::Map<PackedSensorCalibration> output(packed.data() + 11 * sensor);
    output.head<7>() = sensors_from_rig[sensor].params.cast<StorageType>();
    output.tail<4>() =
        Eigen::Map<const InputCalibration>(sensor_calibrations + 4 * sensor)
            .cast<StorageType>();
  }
  return packed;
}

std::vector<StorageType> PackPoses(const std::vector<Rigid3d>& poses) {
  std::vector<StorageType> packed(7 * poses.size());
  for (size_t index = 0; index < poses.size(); ++index) {
    Eigen::Map<PackedPose>(packed.data() + 7 * index) =
        poses[index].params.cast<StorageType>();
  }
  return packed;
}

std::vector<Rigid3d> UnpackPoses(const std::vector<StorageType>& packed) {
  std::vector<Rigid3d> poses(packed.size() / 7);
  for (size_t index = 0; index < poses.size(); ++index) {
    poses[index].params =
        Eigen::Map<const PackedPose>(packed.data() + 7 * index).cast<double>();
    poses[index].rotation().normalize();
  }
  return poses;
}

void TransformPoses(std::vector<Rigid3d>& poses,
                    const Sim3d& transformed_from_source) {
  for (Rigid3d& pose : poses) {
    pose = TransformCameraWorld(transformed_from_source, pose);
  }
}

void TransformPoints(std::vector<StorageType>& points,
                     const Sim3d& transformed_from_source) {
  for (size_t index = 0; index < points.size() / 3; ++index) {
    Eigen::Map<Eigen::Matrix<StorageType, 3, 1>> point(points.data() +
                                                       3 * index);
    point =
        (transformed_from_source * point.cast<double>()).cast<StorageType>();
  }
}

}  // namespace colmap::bundle_adjustment_arrays
