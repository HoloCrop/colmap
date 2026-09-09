// Copyright (c), ETH Zurich and UNC Chapel Hill.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//     * Redistributions of source code must retain the above copyright
//       notice, this list of conditions and the following disclaimer.
//
//     * Redistributions in binary form must reproduce the above copyright
//       notice, this list of conditions and the following disclaimer in the
//       documentation and/or other materials provided with the distribution.
//
//     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
//       its contributors may be used to endorse or promote products derived
//       from this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/util/types.h"

#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <ceres/ceres.h>

namespace colmap {

class Reconstruction;

struct RigCalibrationObservation {
  size_t frame_idx = 0;
  camera_t camera_id = kInvalidCameraId;
  Eigen::Vector2d xy = Eigen::Vector2d::Zero();
};

struct RigCalibrationTrack {
  Eigen::Vector3d xyz = Eigen::Vector3d::Zero();
  std::vector<RigCalibrationObservation> observations;
};

struct RigCalibrationDistancePrior {
  // Metric distance between the first and last rig centers in the group.
  double distance = 0.0;
  double stddev = 1.0;
};

struct RigCalibrationGroup {
  std::vector<Rigid3d> rigs_from_group;
  std::vector<RigCalibrationTrack> tracks;
  RigCalibrationDistancePrior first_to_last_distance;
};

struct RigCalibrationOptions {
  bool refine_focal_length = true;
  bool refine_principal_point = false;
  // Refine all distortion parameters of the camera model.
  bool refine_distortion = true;
  bool refine_sensor_from_rig = true;

  CeresBundleAdjustmentOptions ceres;

  double initialization_angular_stddev_deg = 1.0;
  CeresBundleAdjustmentOptions::LossFunctionType initialization_loss_function_type =
      CeresBundleAdjustmentOptions::LossFunctionType::SOFT_L1;
  double initialization_loss_function_scale = 1.0;

  CeresBundleAdjustmentOptions::LossFunctionType distance_loss_function_type =
      CeresBundleAdjustmentOptions::LossFunctionType::HUBER;
  double distance_loss_function_scale = 1.96;
  // Reject observations above this post-fit pixel reprojection error.
  double max_reprojection_error_pixels = 4.0;
  // Reject tracks whose largest triangulation angle is below this value.
  double min_triangulation_angle_deg = 1.5;
  bool print_summary = true;

  RigCalibrationOptions();

  bool Check() const;
};

struct RigCalibrationObservability {
  // Names follow the tangent-space order of the matrices and vectors below.
  std::vector<std::string> parameter_names;
  Eigen::MatrixXd marginal_information;
  // Moore-Penrose covariance when rank deficient.
  Eigen::MatrixXd marginal_covariance;
  Eigen::VectorXd standard_deviations;
  Eigen::VectorXd normalized_information_eigenvalues;
  size_t rank = 0;
  double normalized_condition_number = std::numeric_limits<double>::infinity();

  bool IsFullRank() const;
};

struct RigCalibrationTiming {
  double group_preparation_seconds = 0.0;
  double local_setup_seconds = 0.0;
  double robust_setup_seconds = 0.0;
  double final_setup_seconds = 0.0;
  double filtering_seconds = 0.0;
  double residual_statistics_seconds = 0.0;
  double observability_seconds = 0.0;
};

struct RigCalibrationSummary : public CeresBundleAdjustmentSummary {
  size_t num_groups = 0;
  size_t num_tracks = 0;
  size_t num_observations = 0;
  size_t num_filtered_groups = 0;
  size_t num_filtered_observations = 0;
  size_t num_invalid_observations = 0;
  double reprojection_rmse = 0.0;
  double distance_prior_rmse = 0.0;
  std::vector<double> reprojection_errors;
  std::vector<double> distance_prior_errors;
  std::vector<ceres::Solver::Summary> stage_summaries;
  RigCalibrationObservability observability;
  RigCalibrationTiming timing;
};

// Jointly calibrates shared camera intrinsics and sensor-from-rig poses from
// independent groups with a uniform frame count of at least two. Group-local
// poses and points are nuisance variables owned by this object and are not
// added to the reconstruction.
class CeresRigCalibrator {
 public:
  CeresRigCalibrator(const RigCalibrationOptions& options,
                     rig_t rig_id,
                     std::vector<RigCalibrationGroup> groups,
                     Reconstruction& reconstruction);
  ~CeresRigCalibrator();

  CeresRigCalibrator(const CeresRigCalibrator&) = delete;
  CeresRigCalibrator& operator=(const CeresRigCalibrator&) = delete;
  CeresRigCalibrator(CeresRigCalibrator&&) = delete;
  CeresRigCalibrator& operator=(CeresRigCalibrator&&) = delete;

  std::shared_ptr<RigCalibrationSummary> Solve();
  std::shared_ptr<ceres::Problem>& Problem();
  const RigCalibrationOptions& Options() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

std::unique_ptr<CeresRigCalibrator> CreateCeresRigCalibrator(
    const RigCalibrationOptions& options,
    rig_t rig_id,
    std::vector<RigCalibrationGroup> groups,
    Reconstruction& reconstruction);

}  // namespace colmap
