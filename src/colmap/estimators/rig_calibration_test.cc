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

#include "colmap/estimators/rig_calibration.h"

#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/scene/synthetic.h"
#include "colmap/sensor/models.h"
#include "colmap/util/testing.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

#include <gtest/gtest.h>

namespace colmap {
namespace {

std::vector<RigCalibrationGroup> BuildGroups(
    const Reconstruction& reconstruction, const size_t group_size = 3) {
  std::vector<frame_t> frame_ids = reconstruction.RegFrameIds();
  std::sort(frame_ids.begin(), frame_ids.end());
  THROW_CHECK_EQ(frame_ids.size() % group_size, 0);

  std::vector<RigCalibrationGroup> groups;
  for (size_t group_begin = 0; group_begin < frame_ids.size();
       group_begin += group_size) {
    RigCalibrationGroup group;
    group.rigs_from_group.reserve(group_size);
    std::map<frame_t, size_t> frame_indices;
    for (size_t frame_idx = 0; frame_idx < group_size; ++frame_idx) {
      const frame_t frame_id = frame_ids[group_begin + frame_idx];
      frame_indices.emplace(frame_id, frame_idx);
      group.rigs_from_group.push_back(
          reconstruction.Frame(frame_id).RigFromWorld());
    }
    group.first_to_last_distance.distance =
        (group.rigs_from_group.back().TgtOriginInSrc() -
         group.rigs_from_group.front().TgtOriginInSrc())
            .norm();
    group.first_to_last_distance.stddev = 1e-3;

    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      RigCalibrationTrack track;
      track.xyz = point3D.xyz;
      for (const TrackElement& track_element : point3D.track.Elements()) {
        const Image& image = reconstruction.Image(track_element.image_id);
        const auto frame_it = frame_indices.find(image.FrameId());
        if (frame_it == frame_indices.end()) {
          continue;
        }
        track.observations.push_back(
            {frame_it->second,
             image.CameraId(),
             image.Point2D(track_element.point2D_idx).xy});
      }
      if (track.observations.size() >= 2) {
        group.tracks.push_back(std::move(track));
      }
    }
    groups.push_back(std::move(group));
  }
  return groups;
}

Reconstruction CreateSyntheticRig(
    const CameraModelId camera_model_id = SimpleRadialCameraModel::model_id,
    const std::vector<double>& camera_params = {1280, 512, 384, 0.05},
    const int num_frames = 9,
    const int num_points = 80) {
  SetPRNGSeed(7);
  Reconstruction reconstruction;
  SyntheticDatasetOptions options;
  options.num_rigs = 1;
  options.num_cameras_per_rig = 3;
  options.num_frames_per_rig = num_frames;
  options.num_points3D = num_points;
  options.num_points2D_without_point3D = 0;
  options.camera_model_id = camera_model_id;
  options.camera_params = camera_params;
  SynthesizeDataset(options, &reconstruction);
  return reconstruction;
}

void ScaleGroupGeometry(const double scale,
                        std::vector<RigCalibrationGroup>* groups) {
  for (RigCalibrationGroup& group : *groups) {
    for (Rigid3d& rig_from_group : group.rigs_from_group) {
      rig_from_group.translation() *= scale;
    }
    for (RigCalibrationTrack& track : group.tracks) {
      track.xyz *= scale;
    }
  }
}

void ScaleGeometry(const double scale,
                   std::vector<RigCalibrationGroup>* groups,
                   Rig* rig) {
  ScaleGroupGeometry(scale, groups);
  for (auto& [sensor_id, sensor_from_rig] : rig->NonRefSensors()) {
    sensor_from_rig->translation() *= scale;
  }
}

TEST(CeresRigCalibrator, RecoversSharedCalibrationAndMetricScale) {
  const Reconstruction ground_truth = CreateSyntheticRig();
  Reconstruction reconstruction = ground_truth;
  std::vector<RigCalibrationGroup> groups = BuildGroups(ground_truth);
  const rig_t rig_id = reconstruction.Rigs().begin()->first;

  ScaleGeometry(1.4, &groups, &reconstruction.Rig(rig_id));
  for (const auto& camera_pair : reconstruction.Cameras()) {
    Camera& camera = reconstruction.Camera(camera_pair.first);
    camera.params[SimpleRadialCameraModel::focal_length_idxs[0]] *= 1.03;
    camera.params[SimpleRadialCameraModel::extra_params_idxs[0]] += 0.01;
  }
  for (auto& [sensor_id, sensor_from_rig] :
       reconstruction.Rig(rig_id).NonRefSensors()) {
    sensor_from_rig->rotation() =
        Eigen::AngleAxisd(DegToRad(1.0), Eigen::Vector3d::UnitZ()) *
        sensor_from_rig->rotation();
  }

  RigCalibrationOptions options;
  options.print_summary = false;
  options.ceres.solver_options.max_num_iterations = 80;
  const std::unique_ptr<CeresRigCalibrator> calibrator =
      CreateCeresRigCalibrator(
          options, rig_id, std::move(groups), reconstruction);
  const std::shared_ptr<RigCalibrationSummary> summary = calibrator->Solve();

  ASSERT_TRUE(summary->IsSolutionUsable());
  ASSERT_EQ(summary->stage_summaries.size(), 3);
  EXPECT_LT(summary->reprojection_rmse, 1e-3);
  EXPECT_LT(summary->distance_prior_rmse, 1e-5);
  EXPECT_EQ(summary->num_filtered_groups, 0);
  EXPECT_EQ(summary->num_filtered_observations, 0);
  EXPECT_TRUE(summary->observability.IsFullRank());

  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    const Camera& expected = ground_truth.Camera(camera_id);
    EXPECT_NEAR(camera.params[SimpleRadialCameraModel::focal_length_idxs[0]],
                expected.params[SimpleRadialCameraModel::focal_length_idxs[0]],
                1e-2);
    EXPECT_NEAR(camera.params[SimpleRadialCameraModel::extra_params_idxs[0]],
                expected.params[SimpleRadialCameraModel::extra_params_idxs[0]],
                1e-4);
  }
  for (const auto& [sensor_id, sensor_from_rig] :
       reconstruction.Rig(rig_id).NonRefSensors()) {
    const Rigid3d& expected = ground_truth.Rig(rig_id).SensorFromRig(sensor_id);
    EXPECT_LT(RadToDeg(sensor_from_rig->rotation().angularDistance(
                  expected.rotation())),
              1e-3);
    EXPECT_LT((sensor_from_rig->translation() - expected.translation()).norm(),
              1e-4);
  }
}

TEST(CeresRigCalibrator, FixedCalibrationEvaluatesNineFrameGroup) {
  const Reconstruction ground_truth = CreateSyntheticRig();
  Reconstruction reconstruction = ground_truth;
  const rig_t rig_id = reconstruction.Rigs().begin()->first;
  std::vector<RigCalibrationGroup> groups = BuildGroups(ground_truth, 9);
  const size_t num_observations =
      std::accumulate(groups.begin(),
                      groups.end(),
                      size_t{0},
                      [](size_t count, const RigCalibrationGroup& group) {
                        for (const RigCalibrationTrack& track : group.tracks) {
                          count += track.observations.size();
                        }
                        return count;
                      });

  RigCalibrationOptions options;
  options.refine_focal_length = false;
  options.refine_principal_point = false;
  options.refine_distortion = false;
  options.refine_sensor_from_rig = false;
  options.print_summary = false;
  const std::unique_ptr<CeresRigCalibrator> calibrator =
      CreateCeresRigCalibrator(
          options, rig_id, std::move(groups), reconstruction);
  const std::shared_ptr<RigCalibrationSummary> summary = calibrator->Solve();

  ASSERT_TRUE(summary->IsSolutionUsable());
  EXPECT_EQ(summary->stage_summaries.size(), 1);
  EXPECT_EQ(summary->reprojection_errors.size(), num_observations);
  EXPECT_LT(summary->reprojection_rmse, 1e-8);
  EXPECT_TRUE(summary->observability.parameter_names.empty());
  EXPECT_TRUE(summary->observability.IsFullRank());
  EXPECT_EQ(reconstruction.Rig(rig_id), ground_truth.Rig(rig_id));
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    EXPECT_EQ(camera, ground_truth.Camera(camera_id));
  }
}

TEST(CeresRigCalibrator, ReportsInvalidHeldOutProjectionAsInfiniteError) {
  const Reconstruction ground_truth = CreateSyntheticRig();
  Reconstruction reconstruction = ground_truth;
  const rig_t rig_id = reconstruction.Rigs().begin()->first;
  const Rig& rig = reconstruction.Rig(rig_id);
  std::vector<RigCalibrationGroup> groups = BuildGroups(ground_truth);

  RigCalibrationGroup& group = groups.front();
  RigCalibrationTrack invalid_track;
  invalid_track.xyz =
      Inverse(group.rigs_from_group[0]) * Eigen::Vector3d(0, 0, -1000);
  for (const sensor_t sensor_id : rig.SensorIds()) {
    const Camera& camera = reconstruction.Camera(sensor_id.id);
    Eigen::Vector3d point_in_cam = group.rigs_from_group[0] * invalid_track.xyz;
    if (!rig.IsRefSensor(sensor_id)) {
      point_in_cam = rig.SensorFromRig(sensor_id) * point_in_cam;
    }
    ASSERT_FALSE(camera.ImgFromCam(point_in_cam).has_value());
    invalid_track.observations.push_back(
        {0, sensor_id.id, Eigen::Vector2d::Zero()});
  }
  const size_t num_invalid_observations = invalid_track.observations.size();
  group.tracks.push_back(std::move(invalid_track));

  RigCalibrationOptions options;
  options.refine_focal_length = false;
  options.refine_principal_point = false;
  options.refine_distortion = false;
  options.refine_sensor_from_rig = false;
  options.print_summary = false;
  const std::unique_ptr<CeresRigCalibrator> calibrator =
      CreateCeresRigCalibrator(
          options, rig_id, std::move(groups), reconstruction);
  const std::shared_ptr<RigCalibrationSummary> summary = calibrator->Solve();

  ASSERT_TRUE(summary->IsSolutionUsable());
  EXPECT_EQ(summary->stage_summaries.size(), 1);
  EXPECT_EQ(summary->num_invalid_observations, num_invalid_observations);
  EXPECT_LT(summary->num_invalid_observations, summary->num_observations);
  EXPECT_EQ(std::count_if(summary->reprojection_errors.begin(),
                          summary->reprojection_errors.end(),
                          [](const double error) { return std::isinf(error); }),
            num_invalid_observations);
  EXPECT_TRUE(std::isinf(summary->reprojection_rmse));
  EXPECT_TRUE(std::isfinite(summary->distance_prior_rmse));
}

TEST(CeresRigCalibrator, PrunesBadObservationBeforeFinalSolve) {
  const Reconstruction ground_truth = CreateSyntheticRig();
  Reconstruction reconstruction = ground_truth;
  const rig_t rig_id = reconstruction.Rigs().begin()->first;
  std::vector<RigCalibrationGroup> groups = BuildGroups(ground_truth);
  const size_t num_observations =
      std::accumulate(groups.begin(),
                      groups.end(),
                      size_t{0},
                      [](size_t count, const RigCalibrationGroup& group) {
                        for (const RigCalibrationTrack& track : group.tracks) {
                          count += track.observations.size();
                        }
                        return count;
                      });
  groups[0].tracks[0].observations[0].xy += Eigen::Vector2d(500, 500);

  RigCalibrationOptions options;
  options.print_summary = false;
  options.ceres.solver_options.max_num_iterations = 80;
  const std::unique_ptr<CeresRigCalibrator> calibrator =
      CreateCeresRigCalibrator(
          options, rig_id, std::move(groups), reconstruction);
  const std::shared_ptr<RigCalibrationSummary> summary = calibrator->Solve();

  ASSERT_TRUE(summary->IsSolutionUsable());
  ASSERT_EQ(summary->stage_summaries.size(), 3);
  EXPECT_EQ(summary->num_filtered_groups, 0);
  EXPECT_EQ(summary->num_filtered_observations, 1);
  EXPECT_EQ(summary->num_observations, num_observations - 1);
  EXPECT_LT(summary->reprojection_rmse, 1e-3);
  EXPECT_TRUE(std::all_of(summary->reprojection_errors.begin(),
                          summary->reprojection_errors.end(),
                          [](const double error) { return error < 4.0; }));
}

TEST(CeresRigCalibrator, OptimizesFullOpenCVDistortion) {
  const std::vector<double> camera_params = {1280,
                                             1260,
                                             512,
                                             384,
                                             0.05,
                                             -0.01,
                                             0.001,
                                             -0.001,
                                             0.002,
                                             0.001,
                                             -0.0005,
                                             0.0002};
  const Reconstruction ground_truth =
      CreateSyntheticRig(FullOpenCVCameraModel::model_id, camera_params);
  Reconstruction reconstruction = ground_truth;
  const rig_t rig_id = reconstruction.Rigs().begin()->first;
  std::vector<RigCalibrationGroup> groups = BuildGroups(ground_truth);
  ScaleGeometry(1.2, &groups, &reconstruction.Rig(rig_id));

  for (const auto& camera_pair : reconstruction.Cameras()) {
    Camera& camera = reconstruction.Camera(camera_pair.first);
    camera.params[0] *= 1.01;
    camera.params[1] *= 0.99;
    camera.params[4] += 0.005;
  }

  RigCalibrationOptions options;
  options.refine_principal_point = true;
  options.print_summary = false;
  options.ceres.solver_options.max_num_iterations = 80;
  const std::unique_ptr<CeresRigCalibrator> calibrator =
      CreateCeresRigCalibrator(
          options, rig_id, std::move(groups), reconstruction);
  const std::shared_ptr<RigCalibrationSummary> summary = calibrator->Solve();

  ASSERT_TRUE(summary->IsSolutionUsable());
  ASSERT_EQ(summary->stage_summaries.size(), 3);
  EXPECT_LT(summary->reprojection_rmse, 1e-3);
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    EXPECT_EQ(camera.model_id, FullOpenCVCameraModel::model_id);
    // Rational coefficients can compensate for one another. Check the fitted
    // image mapping, including regions between the observed tracks.
    const Camera& expected = ground_truth.Camera(camera_id);
    for (int row = 0; row <= 10; ++row) {
      for (int column = 0; column <= 10; ++column) {
        const Eigen::Vector2d pixel(0.5 + (camera.width - 1.0) * column / 10.0,
                                    0.5 + (camera.height - 1.0) * row / 10.0);
        const auto ray = expected.CamFromImg(pixel);
        ASSERT_TRUE(ray.has_value());
        const auto projected = camera.ImgFromCam(ray->homogeneous());
        ASSERT_TRUE(projected.has_value());
        EXPECT_LT((*projected - pixel).norm(), 0.01);
      }
    }
  }
}

TEST(CeresRigCalibrator, ReportsUnobservedRigCameraAsUnobservable) {
  const Reconstruction ground_truth = CreateSyntheticRig();
  Reconstruction reconstruction = ground_truth;
  const rig_t rig_id = reconstruction.Rigs().begin()->first;
  const auto omitted_camera = std::max_element(
      reconstruction.Cameras().begin(),
      reconstruction.Cameras().end(),
      [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });
  const camera_t omitted_camera_id = omitted_camera->first;
  std::vector<RigCalibrationGroup> groups = BuildGroups(ground_truth);
  for (RigCalibrationGroup& group : groups) {
    for (RigCalibrationTrack& track : group.tracks) {
      track.observations.erase(
          std::remove_if(track.observations.begin(),
                         track.observations.end(),
                         [omitted_camera_id](
                             const RigCalibrationObservation& observation) {
                           return observation.camera_id == omitted_camera_id;
                         }),
          track.observations.end());
    }
  }

  RigCalibrationOptions options;
  options.print_summary = false;
  const std::unique_ptr<CeresRigCalibrator> calibrator =
      CreateCeresRigCalibrator(
          options, rig_id, std::move(groups), reconstruction);
  const std::shared_ptr<RigCalibrationSummary> summary = calibrator->Solve();

  ASSERT_TRUE(summary->IsSolutionUsable());
  EXPECT_FALSE(summary->observability.IsFullRank());
  const std::string omitted_prefix =
      "camera[" + std::to_string(omitted_camera_id) + "].";
  EXPECT_TRUE(std::any_of(summary->observability.parameter_names.begin(),
                          summary->observability.parameter_names.end(),
                          [&omitted_prefix](const std::string& name) {
                            return name.rfind(omitted_prefix, 0) == 0;
                          }));
}

}  // namespace
}  // namespace colmap
