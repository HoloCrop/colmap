#pragma once

#include "colmap/estimators/rig_calibration.h"

namespace colmap {

struct RigCalibrationPosePair {
  size_t frame_idx1;
  camera_t camera_id1;
  size_t frame_idx2;
  camera_t camera_id2;
  Rigid3d cam2_from_cam1;
};

struct RigCalibrationInitializationGroup {
  std::vector<Rigid3d> rigs_from_group;
  std::vector<RigCalibrationPosePair> pairs;
  RigCalibrationDistancePrior first_to_last_distance;
};

struct RigCalibrationInitializationSummary {
  ceres::Solver::Summary rotation_summary;
  ceres::Solver::Summary position_summary;
  size_t rotation_rank = 0;
  size_t num_rotation_parameters = 0;
  size_t position_rank = 0;
  size_t num_position_parameters = 0;
};

// Initialize shared camera extrinsics and group-local frame poses from
// calibrated two-view poses, without scene points. Input poses are seeds;
// endpoint distances provide the only metric measurements. The reference
// camera and each group's first frame fix the coordinate gauges. Input poses
// are updated only after both solves succeed and camera positions are
// observable after marginalizing group-local frame centers.
RigCalibrationInitializationSummary InitializeRigCalibration(
    const RigCalibrationOptions& options,
    rig_t rig_id,
    std::vector<RigCalibrationInitializationGroup>& groups,
    Reconstruction& reconstruction);

}  // namespace colmap
