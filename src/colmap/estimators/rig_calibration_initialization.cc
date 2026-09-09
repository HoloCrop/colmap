#include "colmap/estimators/rig_calibration_initialization.h"

#include "colmap/estimators/cost_functions/manifold.h"
#include "colmap/estimators/rig_calibration_observability.h"
#include "colmap/scene/reconstruction.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/threading.h"

#include <algorithm>
#include <array>
#include <string>
#include <utility>

#include <Eigen/Eigenvalues>
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/rotation.h>

namespace colmap {
namespace {

constexpr double kAngularStddev = EIGEN_PI / 180.0;
constexpr double kRankTolerance = 1e-10;

// Store centers separately so changing a rotation cannot move its camera.
struct PoseParameters {
  explicit PoseParameters(const Rigid3d& pose)
      : rotation(pose.rotation()), center(pose.TgtOriginInSrc()) {}

  Rigid3d Pose() const { return Rigid3d(rotation, -(rotation * center)); }

  Eigen::Quaterniond rotation;
  Eigen::Vector3d center;
};

// Same-frame and same-camera pairs share parameter blocks. Each Ceres factor
// receives the unique blocks and maps its four semantic roles onto them.
struct PairParameterBlocks {
  explicit PairParameterBlocks(const std::array<double*, 4>& roles) {
    for (size_t index = 0; index < roles.size(); ++index) {
      const auto found = std::find(blocks.begin(), blocks.end(), roles[index]);
      slots[index] = std::distance(blocks.begin(), found);
      if (found == blocks.end()) {
        blocks.push_back(roles[index]);
      }
    }
  }

  std::array<int, 4> slots;
  std::vector<double*> blocks;
};

struct RelativeRotationCost {
  std::array<int, 4> slots;
  Eigen::Quaterniond camera2_from_camera1;

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    const Eigen::Quaternion<T> camera1_from_group =
        Eigen::Map<const Eigen::Quaternion<T>>(parameters[slots[0]]) *
        Eigen::Map<const Eigen::Quaternion<T>>(parameters[slots[1]]);
    const Eigen::Quaternion<T> camera2_from_group =
        Eigen::Map<const Eigen::Quaternion<T>>(parameters[slots[2]]) *
        Eigen::Map<const Eigen::Quaternion<T>>(parameters[slots[3]]);
    const Eigen::Quaternion<T> error =
        camera2_from_camera1.conjugate().cast<T>() * camera2_from_group *
        camera1_from_group.conjugate();
    const T quaternion[] = {error.w(), error.x(), error.y(), error.z()};
    ceres::QuaternionToAngleAxis(quaternion, residuals);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> result(residuals);
    result /= T(kAngularStddev);
    return true;
  }
};

struct RelativeDirectionCost {
  std::array<int, 4> slots;
  Eigen::Matrix3d group_from_rig1;
  Eigen::Matrix3d group_from_rig2;
  Eigen::Vector3d direction_in_group;

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    const Eigen::Matrix<T, 3, 1> displacement =
        group_from_rig1.cast<T>() *
            Eigen::Map<const Eigen::Matrix<T, 3, 1>>(parameters[slots[0]]) +
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(parameters[slots[1]]) -
        group_from_rig2.cast<T>() *
            Eigen::Map<const Eigen::Matrix<T, 3, 1>>(parameters[slots[2]]) -
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(parameters[slots[3]]);
    Eigen::Map<Eigen::Matrix<T, 3, 1>> result(residuals);
    result = (displacement.normalized() - direction_in_group.cast<T>()) /
             T(kAngularStddev);
    return true;
  }
};

struct EndpointDistanceCost {
  double distance;
  double stddev;

  template <typename T>
  bool operator()(const T* const first_center,
                  const T* const last_center,
                  T* residuals) const {
    const Eigen::Matrix<T, 3, 1> displacement =
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(last_center) -
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(first_center);
    residuals[0] = (displacement.norm() - T(distance)) / T(stddev);
    return true;
  }
};

template <typename Functor>
ceres::ResidualBlockId AddPairResidual(ceres::Problem& problem,
                                       Functor functor,
                                       const PairParameterBlocks& parameters,
                                       const int block_size,
                                       ceres::LossFunction* loss) {
  auto* cost = new ceres::DynamicAutoDiffCostFunction<Functor>(
      new Functor(std::move(functor)));
  for (size_t index = 0; index < parameters.blocks.size(); ++index) {
    cost->AddParameterBlock(block_size);
  }
  cost->SetNumResiduals(3);
  return problem.AddResidualBlock(cost, loss, parameters.blocks);
}

class RigCalibrationInitializer {
 public:
  RigCalibrationInitializer(
      const RigCalibrationOptions& options,
      const Rig& rig,
      const std::vector<RigCalibrationInitializationGroup>& groups)
      : options_(options), reference_camera_id_(rig.RefSensorId().id) {
    THROW_CHECK(options.Check());
    THROW_CHECK_EQ(rig.RefSensorId().type, SensorType::CAMERA);
    THROW_CHECK_GT(rig.NumSensors(), 1);
    THROW_CHECK_GT(groups.size(), 0);
    for (const sensor_t sensor_id : rig.SensorIds()) {
      THROW_CHECK_EQ(sensor_id.type, SensorType::CAMERA);
      camera_ids_.push_back(sensor_id.id);
      cameras_.emplace(sensor_id.id,
                       PoseParameters(rig.IsRefSensor(sensor_id)
                                          ? Rigid3d()
                                          : rig.SensorFromRig(sensor_id)));
    }
    for (const auto& group : groups) {
      THROW_CHECK_GE(group.rigs_from_group.size(), 2);
      THROW_CHECK_GT(group.first_to_last_distance.distance, 0);
      THROW_CHECK_GT(group.first_to_last_distance.stddev, 0);
      std::vector<PoseParameters> frames;
      frames.reserve(group.rigs_from_group.size());
      for (const Rigid3d& pose : group.rigs_from_group) {
        frames.emplace_back(pose);
      }
      frames_.push_back(std::move(frames));
    }
  }

  void InitializeRotations(
      const std::vector<RigCalibrationInitializationGroup>& groups,
      RigCalibrationInitializationSummary& summary) {
    ceres::SoftLOneLoss loss(1.0);
    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problem_options);
    const auto add_rotation = [&](PoseParameters& pose) {
      double* quaternion = pose.rotation.coeffs().data();
      problem.AddParameterBlock(quaternion, 4);
      SetManifold(&problem, quaternion, CreateEigenQuaternionManifold());
    };
    std::vector<double*> global_blocks;
    for (const camera_t camera_id : camera_ids_) {
      add_rotation(cameras_.at(camera_id));
      if (camera_id != reference_camera_id_) {
        global_blocks.push_back(
            cameras_.at(camera_id).rotation.coeffs().data());
      }
    }
    problem.SetParameterBlockConstant(
        cameras_.at(reference_camera_id_).rotation.coeffs().data());
    std::vector<internal::RigCalibrationObservabilityGroupData> group_data(
        groups.size());
    for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
      auto& frames = frames_[group_index];
      auto& data = group_data[group_index];
      for (auto& frame : frames) {
        add_rotation(frame);
      }
      for (size_t frame_index = 1; frame_index < frames.size(); ++frame_index) {
        data.local_pose_blocks.push_back(
            frames[frame_index].rotation.coeffs().data());
      }
      problem.SetParameterBlockConstant(
          frames.front().rotation.coeffs().data());
      for (const auto& pair : groups[group_index].pairs) {
        const PairParameterBlocks parameters({
            cameras_.at(pair.camera_id1).rotation.coeffs().data(),
            frames.at(pair.frame_idx1).rotation.coeffs().data(),
            cameras_.at(pair.camera_id2).rotation.coeffs().data(),
            frames.at(pair.frame_idx2).rotation.coeffs().data(),
        });
        data.non_track_residual_blocks.push_back(AddPairResidual(
            problem,
            RelativeRotationCost{parameters.slots,
                                 pair.cam2_from_cam1.rotation()},
            parameters,
            4,
            &loss));
      }
    }
    Solve(problem, summary.rotation_summary);
    THROW_CHECK(summary.rotation_summary.IsSolutionUsable())
        << "Two-view rotation initialization failed: "
        << summary.rotation_summary.BriefReport();
    summary.rotation_rank = CheckObservability(
        problem, global_blocks, group_data, "camera rotations");
    summary.num_rotation_parameters = 3 * global_blocks.size();
  }

  void InitializePositions(
      const std::vector<RigCalibrationInitializationGroup>& groups,
      RigCalibrationInitializationSummary& summary) {
    ceres::SoftLOneLoss loss(1.0);
    CeresBundleAdjustmentOptions distance_options = options_.ceres;
    distance_options.loss_function_type = options_.distance_loss_function_type;
    distance_options.loss_function_scale =
        options_.distance_loss_function_scale;
    const auto distance_loss = distance_options.CreateLossFunction();
    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem(problem_options);
    std::vector<double*> global_blocks;
    for (const camera_t camera_id : camera_ids_) {
      double* center = cameras_.at(camera_id).center.data();
      problem.AddParameterBlock(center, 3);
      if (camera_id == reference_camera_id_) {
        problem.SetParameterBlockConstant(center);
      } else {
        global_blocks.push_back(center);
      }
    }
    std::vector<internal::RigCalibrationObservabilityGroupData> group_data(
        groups.size());
    for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
      auto& frames = frames_[group_index];
      auto& data = group_data[group_index];
      for (size_t frame_index = 0; frame_index < frames.size(); ++frame_index) {
        double* center = frames[frame_index].center.data();
        problem.AddParameterBlock(center, 3);
        if (frame_index == 0) {
          problem.SetParameterBlockConstant(center);
        } else {
          data.local_pose_blocks.push_back(center);
        }
      }
      for (const auto& pair : groups[group_index].pairs) {
        auto& camera1 = cameras_.at(pair.camera_id1);
        auto& camera2 = cameras_.at(pair.camera_id2);
        auto& frame1 = frames.at(pair.frame_idx1);
        auto& frame2 = frames.at(pair.frame_idx2);
        const PairParameterBlocks parameters({camera1.center.data(),
                                              frame1.center.data(),
                                              camera2.center.data(),
                                              frame2.center.data()});
        const Eigen::Vector3d direction =
            (camera2.rotation * frame2.rotation).conjugate() *
            pair.cam2_from_cam1.translation().normalized();
        data.non_track_residual_blocks.push_back(
            AddPairResidual(problem,
                            RelativeDirectionCost{
                                parameters.slots,
                                frame1.rotation.conjugate().toRotationMatrix(),
                                frame2.rotation.conjugate().toRotationMatrix(),
                                direction},
                            parameters,
                            3,
                            &loss));
      }
      const auto& prior = groups[group_index].first_to_last_distance;
      data.non_track_residual_blocks.push_back(problem.AddResidualBlock(
          new ceres::AutoDiffCostFunction<EndpointDistanceCost, 1, 3, 3>(
              new EndpointDistanceCost{prior.distance, prior.stddev}),
          distance_loss.get(),
          frames.front().center.data(),
          frames.back().center.data()));
    }
    Solve(problem, summary.position_summary);
    THROW_CHECK(summary.position_summary.IsSolutionUsable())
        << "Two-view position initialization failed: "
        << summary.position_summary.BriefReport();
    summary.position_rank = CheckObservability(
        problem, global_blocks, group_data, "metric camera positions");
    summary.num_position_parameters = 3 * global_blocks.size();
  }

  void StoreResults(std::vector<RigCalibrationInitializationGroup>& groups,
                    Rig& rig) const {
    for (const camera_t camera_id : camera_ids_) {
      if (camera_id != reference_camera_id_) {
        rig.SetSensorFromRig(sensor_t(SensorType::CAMERA, camera_id),
                             cameras_.at(camera_id).Pose());
      }
    }
    for (size_t group_index = 0; group_index < groups.size(); ++group_index) {
      auto& group = groups[group_index];
      for (size_t frame_index = 0; frame_index < frames_[group_index].size();
           ++frame_index) {
        group.rigs_from_group[frame_index] =
            frames_[group_index][frame_index].Pose();
      }
    }
  }

 private:
  size_t CheckObservability(
      const ceres::Problem& problem,
      const std::vector<double*>& global_blocks,
      const std::vector<internal::RigCalibrationObservabilityGroupData>&
          group_data,
      const std::string& description) const {
    const Eigen::MatrixXd information =
        internal::ComputeRigCalibrationMarginalInformation(
            problem,
            global_blocks,
            group_data,
            kRankTolerance,
            options_.ceres.solver_options.num_threads);
    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(information);
    THROW_CHECK_EQ(eigen.info(), Eigen::Success);
    const double threshold = kRankTolerance * eigen.eigenvalues().maxCoeff();
    const size_t rank = (eigen.eigenvalues().array() > threshold).count();
    THROW_CHECK_EQ(rank, information.rows())
        << "Two-view poses do not constrain " << description;
    return rank;
  }

  void Solve(ceres::Problem& problem, ceres::Solver::Summary& summary) const {
    ceres::Solver::Options solver_options = options_.ceres.solver_options;
    solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
    solver_options.num_threads =
        GetEffectiveNumThreads(solver_options.num_threads);
    ceres::Solve(solver_options, &problem, &summary);
  }

  const RigCalibrationOptions& options_;
  camera_t reference_camera_id_;
  std::vector<camera_t> camera_ids_;
  NodeHashMap<camera_t, PoseParameters> cameras_;
  std::vector<std::vector<PoseParameters>> frames_;
};

}  // namespace

RigCalibrationInitializationSummary InitializeRigCalibration(
    const RigCalibrationOptions& options,
    const rig_t rig_id,
    std::vector<RigCalibrationInitializationGroup>& groups,
    Reconstruction& reconstruction) {
  Rig& rig = reconstruction.Rig(rig_id);
  RigCalibrationInitializer initializer(options, rig, groups);
  RigCalibrationInitializationSummary summary;
  initializer.InitializeRotations(groups, summary);
  initializer.InitializePositions(groups, summary);
  initializer.StoreResults(groups, rig);
  return summary;
}

}  // namespace colmap
