#include "colmap/estimators/rig_calibration_initialization.h"

#include "colmap/math/math.h"
#include "colmap/scene/reconstruction.h"

#include <array>
#include <cmath>
#include <map>
#include <random>
#include <stdexcept>
#include <utility>

#include <gtest/gtest.h>

namespace colmap {
namespace {

constexpr rig_t kRigId = 1;
constexpr camera_t kReferenceCameraId = 2;
constexpr std::array<camera_t, 4> kCameraIds = {1, 2, 3, 12};

Eigen::Quaterniond RotationDegrees(const Eigen::Vector3d& degrees) {
  return Eigen::AngleAxisd(DegToRad(degrees.x()), Eigen::Vector3d::UnitX()) *
         Eigen::AngleAxisd(DegToRad(degrees.y()), Eigen::Vector3d::UnitY()) *
         Eigen::AngleAxisd(DegToRad(degrees.z()), Eigen::Vector3d::UnitZ());
}

Rigid3d PoseFromCenter(const Eigen::Quaterniond& rotation,
                       const Eigen::Vector3d& center) {
  return Rigid3d(rotation, -(rotation * center));
}

struct InitializationFixture {
  Reconstruction reconstruction;
  std::vector<RigCalibrationInitializationGroup> groups;
  std::map<camera_t, Rigid3d> expected_cameras;
};

InitializationFixture CreateFixture(const bool collinear_motion = false) {
  InitializationFixture fixture;
  const std::array<Eigen::Vector3d, 4> centers = {
      Eigen::Vector3d(-0.22, 0.005, 0.003),
      Eigen::Vector3d::Zero(),
      Eigen::Vector3d(0.24, 0.01, -0.006),
      Eigen::Vector3d(0.46, 0.015, 0.004)};
  const std::array<Eigen::Vector3d, 4> angles = {Eigen::Vector3d(-3, 5, 4),
                                                 Eigen::Vector3d::Zero(),
                                                 Eigen::Vector3d(4, 0, -3),
                                                 Eigen::Vector3d(5, 31, 4)};
  Rig rig;
  rig.SetRigId(kRigId);
  rig.AddRefSensor(sensor_t(SensorType::CAMERA, kReferenceCameraId));
  for (size_t index = 0; index < kCameraIds.size(); ++index) {
    const camera_t camera_id = kCameraIds[index];
    fixture.reconstruction.AddCamera(
        Camera::CreateFromModelName(camera_id, "PINHOLE", 500, 640, 480));
    Eigen::Vector3d center = centers[index];
    if (collinear_motion) {
      center.y() = 0;
      center.z() = 0;
    }
    const Rigid3d pose = PoseFromCenter(RotationDegrees(angles[index]), center);
    fixture.expected_cameras.emplace(camera_id, pose);
    if (camera_id != kReferenceCameraId) {
      rig.AddSensor(sensor_t(SensorType::CAMERA, camera_id), pose);
    }
  }
  fixture.reconstruction.AddRig(rig);
  std::mt19937 random(17);
  std::uniform_real_distribution<double> scale(0.1, 9.0);
  size_t pair_count = 0;
  for (size_t group_index = 0; group_index < 8; ++group_index) {
    RigCalibrationInitializationGroup group;
    for (size_t frame_index = 0; frame_index < 3; ++frame_index) {
      const double step = static_cast<double>(frame_index);
      const Eigen::Vector3d center =
          collinear_motion
              ? Eigen::Vector3d(0.15 * step, 0, 0)
              : Eigen::Vector3d(
                    0.01 * group_index * step, 0.15 * step, 0.025 * step);
      const Eigen::Quaterniond rotation =
          collinear_motion
              ? Eigen::Quaterniond::Identity()
              : RotationDegrees(Eigen::Vector3d(0.5 * step, -2 * step, step));
      group.rigs_from_group.push_back(PoseFromCenter(rotation, center));
    }
    group.first_to_last_distance.distance =
        group.rigs_from_group.back().TgtOriginInSrc().norm();
    group.first_to_last_distance.stddev = std::sqrt(2.0) * 0.025;
    for (size_t image1 = 0; image1 < 12; ++image1) {
      for (size_t image2 = image1 + 1; image2 < 12; ++image2) {
        const size_t frame1 = image1 / kCameraIds.size();
        const size_t frame2 = image2 / kCameraIds.size();
        if (frame2 - frame1 > 1) {
          continue;
        }
        const camera_t camera1 = kCameraIds[image1 % kCameraIds.size()];
        const camera_t camera2 = kCameraIds[image2 % kCameraIds.size()];
        const Rigid3d pose1 = fixture.expected_cameras.at(camera1) *
                              group.rigs_from_group[frame1];
        const Rigid3d pose2 = fixture.expected_cameras.at(camera2) *
                              group.rigs_from_group[frame2];
        Rigid3d relative = pose2 * Inverse(pose1);
        relative.translation() =
            relative.translation().normalized() * scale(random);
        RigCalibrationPosePair pair{frame1, camera1, frame2, camera2, relative};
        if (++pair_count % 2 == 0) {
          std::swap(pair.frame_idx1, pair.frame_idx2);
          std::swap(pair.camera_id1, pair.camera_id2);
          pair.cam2_from_cam1 = Inverse(pair.cam2_from_cam1);
        }
        group.pairs.push_back(pair);
      }
    }
    fixture.groups.push_back(std::move(group));
  }
  return fixture;
}

void PerturbCamera(InitializationFixture& fixture,
                   const camera_t camera_id,
                   const double rotation_degrees = -32) {
  Rigid3d& pose = fixture.reconstruction.Rig(kRigId).SensorFromRig(
      sensor_t(SensorType::CAMERA, camera_id));
  pose = PoseFromCenter(
      RotationDegrees(Eigen::Vector3d(12, rotation_degrees, 9)) *
          pose.rotation(),
      pose.TgtOriginInSrc() + Eigen::Vector3d(-0.085, 0.07, -0.03));
}

void PerturbFrames(std::vector<RigCalibrationInitializationGroup>& groups) {
  for (auto& group : groups) {
    for (size_t frame_index = 1; frame_index < group.rigs_from_group.size();
         ++frame_index) {
      Rigid3d& pose = group.rigs_from_group[frame_index];
      pose = PoseFromCenter(
          RotationDegrees(Eigen::Vector3d(-8, 5, 6)) * pose.rotation(),
          1.25 * pose.TgtOriginInSrc() + Eigen::Vector3d(0.04, -0.02, 0.06));
    }
  }
}

void AddOutliers(std::vector<RigCalibrationInitializationGroup>& groups) {
  std::mt19937 random(29);
  std::normal_distribution<double> normal;
  size_t pair_count = 0;
  for (auto& group : groups) {
    for (auto& pair : group.pairs) {
      if (++pair_count % 29 == 0) {
        pair.cam2_from_cam1.rotation() =
            RotationDegrees(Eigen::Vector3d(40, -80, 35)) *
            pair.cam2_from_cam1.rotation();
        pair.cam2_from_cam1.translation() =
            Eigen::Vector3d(normal(random), normal(random), normal(random));
      }
    }
  }
}

RigCalibrationOptions InitializationOptions() {
  RigCalibrationOptions options;
  options.print_summary = false;
  options.ceres.solver_options.num_threads = 1;
  options.ceres.solver_options.max_num_iterations = 200;
  return options;
}

void ExpectRecoveredCameras(const InitializationFixture& fixture,
                            const double position_tolerance,
                            const double angle_tolerance_degrees) {
  for (const auto& [camera_id, expected] : fixture.expected_cameras) {
    if (camera_id == kReferenceCameraId) {
      continue;
    }
    SCOPED_TRACE(camera_id);
    const Rigid3d& actual = fixture.reconstruction.Rig(kRigId).SensorFromRig(
        sensor_t(SensorType::CAMERA, camera_id));
    EXPECT_LT((actual.TgtOriginInSrc() - expected.TgtOriginInSrc()).norm(),
              position_tolerance);
    EXPECT_LT(RadToDeg(actual.rotation().angularDistance(expected.rotation())),
              angle_tolerance_degrees);
  }
}

class RigCalibrationInitializationRotation
    : public testing::TestWithParam<double> {};

TEST_P(RigCalibrationInitializationRotation,
       RecoversCoherentCameraMotionAndUncertainGroupPoseSeeds) {
  auto fixture = CreateFixture();
  const auto expected_groups = fixture.groups;
  PerturbCamera(fixture, 12, GetParam());
  PerturbFrames(fixture.groups);

  InitializeRigCalibration(
      InitializationOptions(), kRigId, fixture.groups, fixture.reconstruction);

  ExpectRecoveredCameras(fixture, 1e-4, 1e-3);
  for (size_t group_index = 0; group_index < fixture.groups.size();
       ++group_index) {
    const auto& actual = fixture.groups[group_index];
    const auto& expected = expected_groups[group_index];
    for (size_t frame_index = 0; frame_index < actual.rigs_from_group.size();
         ++frame_index) {
      EXPECT_LT((actual.rigs_from_group[frame_index].TgtOriginInSrc() -
                 expected.rigs_from_group[frame_index].TgtOriginInSrc())
                    .norm(),
                1e-4);
      EXPECT_LT(
          RadToDeg(
              actual.rigs_from_group[frame_index].rotation().angularDistance(
                  expected.rigs_from_group[frame_index].rotation())),
          1e-3);
    }
    EXPECT_DOUBLE_EQ(actual.first_to_last_distance.distance,
                     expected.first_to_last_distance.distance);
  }
}

INSTANTIATE_TEST_SUITE_P(LargeCameraMotion,
                         RigCalibrationInitializationRotation,
                         testing::Values(-32.0, -120.0));

TEST(RigCalibrationInitialization, RecoversSeveralMovedCamerasDespiteOutliers) {
  auto fixture = CreateFixture();
  for (const camera_t camera_id : {1, 3, 12}) {
    PerturbCamera(fixture, camera_id);
  }
  PerturbFrames(fixture.groups);
  AddOutliers(fixture.groups);

  InitializeRigCalibration(
      InitializationOptions(), kRigId, fixture.groups, fixture.reconstruction);

  ExpectRecoveredCameras(fixture, 0.004, 0.3);
}

TEST(RigCalibrationInitialization, RejectsUnobservableCollinearBaselines) {
  auto fixture = CreateFixture(true);
  const auto original_groups = fixture.groups;
  const Rig original_rig = fixture.reconstruction.Rig(kRigId);

  try {
    InitializeRigCalibration(InitializationOptions(),
                             kRigId,
                             fixture.groups,
                             fixture.reconstruction);
    FAIL() << "Collinear baselines must not be reported as observable";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what()).find("metric camera positions"),
              std::string::npos);
  }

  EXPECT_EQ(fixture.reconstruction.Rig(kRigId), original_rig);
  for (size_t group_index = 0; group_index < fixture.groups.size();
       ++group_index) {
    const auto& actual = fixture.groups[group_index].rigs_from_group;
    const auto& original = original_groups[group_index].rigs_from_group;
    for (size_t frame_index = 0; frame_index < actual.size(); ++frame_index) {
      EXPECT_EQ(actual[frame_index].params, original[frame_index].params);
    }
  }
}

}  // namespace
}  // namespace colmap
