#pragma once

#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/geometry/sim3.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace colmap::bundle_adjustment_arrays {

struct NormalizedProblem {
  Sim3d normalized_from_metric;
  std::vector<Rigid3d> rigs_from_world;
  std::vector<Rigid3d> sensors_from_rig;
  std::vector<StorageType> points;
};

Sim3d ComputeNormalization(const std::vector<Rigid3d>& rigs_from_world,
                           const uint32_t* image_frame_indices,
                           const uint32_t* image_sensor_indices,
                           size_t num_images,
                           const std::vector<Rigid3d>& sensors_from_rig);

Sim3d ComputeNormalization(const std::vector<Rigid3d>& rigs_from_world,
                           const uint32_t* image_frame_indices,
                           const uint32_t* image_sensor_indices,
                           const uint32_t* selected_image_indices,
                           size_t num_selected_images,
                           const std::vector<Rigid3d>& sensors_from_rig);

NormalizedProblem NormalizeProblem(
    const Sim3d& normalized_from_metric,
    const std::vector<Rigid3d>& initial_rigs_from_world,
    const float* initial_points,
    size_t num_points,
    const std::vector<Rigid3d>& sensors_from_rig);

// Packs each sensor's sensor_from_rig storage followed by its PINHOLE
// [fx, fy, cx, cy] calibration (4 floats per sensor) for the row factor.
std::vector<StorageType> PackSensorCalibrations(
    const std::vector<Rigid3d>& sensors_from_rig,
    const float* sensor_calibrations);

std::vector<StorageType> PackPoses(const std::vector<Rigid3d>& poses);
std::vector<Rigid3d> UnpackPoses(const std::vector<StorageType>& packed);

void TransformPoses(std::vector<Rigid3d>& poses,
                    const Sim3d& transformed_from_source);
void TransformPoints(std::vector<StorageType>& points,
                     const Sim3d& transformed_from_source);

}  // namespace colmap::bundle_adjustment_arrays
