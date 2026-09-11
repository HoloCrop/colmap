#include "colmap/estimators/fixed_rig_global_positioning.h"

#include "colmap/math/math.h"
#include "colmap/math/random.h"
#include "colmap/util/cuda.h"
#include "colmap/util/misc.h"
#include "colmap/util/threading.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <queue>
#include <tuple>

#include <Eigen/Cholesky>
#include <Eigen/SVD>

namespace colmap {
namespace {

Eigen::Vector3d RandomVector3d(double low, double high) {
  return Eigen::Vector3d(RandomUniformReal(low, high),
                         RandomUniformReal(low, high),
                         RandomUniformReal(low, high));
}

struct FixedRigPairwiseDirectionCostFunctor {
  FixedRigPairwiseDirectionCostFunctor(
      const Eigen::Vector3d& cam_from_point3D_direction,
      const Eigen::Vector3d& cam_from_rig_translation)
      : cam_from_point3D_direction_(cam_from_point3D_direction),
        cam_from_rig_translation_(cam_from_rig_translation) {}

  template <typename T>
  bool operator()(const T* point3D, const T* rig_in_world, T* residuals) const {
    const Eigen::Matrix<T, 3, 1> displacement =
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(point3D) -
        Eigen::Map<const Eigen::Matrix<T, 3, 1>>(rig_in_world) +
        cam_from_rig_translation_.cast<T>();
    const Eigen::Matrix<T, 3, 1> bearing =
        cam_from_point3D_direction_.cast<T>();
    const T unconstrained_scale =
        bearing.dot(displacement) / displacement.squaredNorm();
    const T scale =
        unconstrained_scale < T(1e-5) ? T(1e-5) : unconstrained_scale;
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vector(residuals);
    residuals_vector = bearing - scale * displacement;
    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& cam_from_point3D_direction,
      const Eigen::Vector3d& cam_from_rig_translation) {
    return new ceres::
        AutoDiffCostFunction<FixedRigPairwiseDirectionCostFunctor, 3, 3, 3>(
            new FixedRigPairwiseDirectionCostFunctor(cam_from_point3D_direction,
                                                     cam_from_rig_translation));
  }

  const Eigen::Vector3d cam_from_point3D_direction_;
  const Eigen::Vector3d cam_from_rig_translation_;
};

struct FixedRigFramePositionCostFunctor {
  FixedRigFramePositionCostFunctor(const Eigen::Vector3d& center2_from_center1,
                                   double weight)
      : center2_from_center1_(center2_from_center1), weight_(weight) {}

  template <typename T>
  bool operator()(const T* center1, const T* center2, T* residuals) const {
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals_vector(residuals);
    residuals_vector =
        T(weight_) * (Eigen::Map<const Eigen::Matrix<T, 3, 1>>(center2) -
                      Eigen::Map<const Eigen::Matrix<T, 3, 1>>(center1) -
                      center2_from_center1_.cast<T>());
    return true;
  }

  static ceres::CostFunction* Create(
      const Eigen::Vector3d& center2_from_center1, double weight) {
    return new ceres::
        AutoDiffCostFunction<FixedRigFramePositionCostFunctor, 3, 3, 3>(
            new FixedRigFramePositionCostFunctor(center2_from_center1, weight));
  }

  const Eigen::Vector3d center2_from_center1_;
  const double weight_;
};

}  // namespace

FixedRigGlobalPositioner::FixedRigGlobalPositioner(
    const GlobalPositionerOptions& options,
    const FixedRigGlobalPositionerOptions& rig_options,
    double min_tri_angle_deg)
    : options_(options),
      rig_options_(rig_options),
      min_tri_angle_deg_(min_tri_angle_deg) {
  if (options_.random_seed >= 0) {
    SetPRNGSeed(static_cast<unsigned>(options_.random_seed));
  }
}

bool FixedRigGlobalPositioner::Solve(
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors) {
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Number of images = " << reconstruction.NumImages();
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Number of tracks = " << reconstruction.NumPoints3D();
    return false;
  }

  options_.solver_options.num_threads =
      GetEffectiveNumThreads(options_.solver_options.num_threads);
  SetupProblem();
  InitializePositions(pose_graph, reconstruction, pose_priors);
  ConfigureSolver(reconstruction);

  if (options_.optimize_positions && options_.optimize_points) {
    std::vector<FrameConstraint> constraints;
    if (BuildFrameConstraints(reconstruction, constraints)) {
      LOG(INFO) << "Built " << constraints.size()
                << " calibrated-rig frame constraints";
      return SolveFramePositions(reconstruction, constraints);
    }
    if (rig_options_.require_frame_constraints) {
      LOG(ERROR) << "Calibrated-rig frame constraints do not cover all frames";
      return false;
    }
    LOG(INFO) << "Calibrated-rig frame constraints do not cover all frames; "
                 "using observation-level positioning";
  }

  if (initialized_from_pose_priors_) {
    ConvertBackResults(reconstruction);
    RefinePoints(reconstruction);
  }

  Timer stage_timer;
  stage_timer.Start();
  AddPointConstraints(reconstruction);
  LOG(INFO) << "Fixed-rig residual construction done in "
            << stage_timer.ElapsedSeconds() << " seconds ("
            << problem_->NumResidualBlocks() << " residuals, "
            << problem_->NumParameterBlocks() << " parameter blocks)";

  if (options_.use_parameter_block_ordering) {
    AddParameterBlockOrdering(reconstruction);
  }
  ParameterizeVariables(reconstruction);

  options_.solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solver::Summary summary;
  ceres::Solve(options_.solver_options, problem_.get(), &summary);
  LOG(INFO) << (VLOG_IS_ON(2) ? summary.FullReport() : summary.BriefReport());

  ConvertBackResults(reconstruction);
  RefinePoints(reconstruction);
  return summary.IsSolutionUsable();
}

void FixedRigGlobalPositioner::SetupProblem() {
  ceres::Problem::Options problem_options;
  problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
  problem_ = std::make_unique<ceres::Problem>(problem_options);
  calibrated_loss_function_ = options_.CreateLossFunction();
  uncalibrated_loss_function_ = std::make_shared<ceres::ScaledLoss>(
      calibrated_loss_function_.get(), 0.5, ceres::DO_NOT_TAKE_OWNERSHIP);
  loss_function_ = calibrated_loss_function_;
  frame_centers_.clear();
}

FlatHashSet<frame_t> FixedRigGlobalPositioner::ConstrainedFrameIds(
    const PoseGraph& pose_graph, const Reconstruction& reconstruction) const {
  FlatHashSet<frame_t> frame_ids;
  frame_ids.reserve(reconstruction.NumFrames());
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    frame_ids.insert(reconstruction.Image(image_id1).FrameId());
    frame_ids.insert(reconstruction.Image(image_id2).FrameId());
  }
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }
    for (const TrackElement& observation : point3D.track.Elements()) {
      const Image& image = reconstruction.Image(observation.image_id);
      if (image.HasPose()) {
        frame_ids.insert(image.FrameId());
      }
    }
  }
  return frame_ids;
}

void FixedRigGlobalPositioner::InitializePositions(
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors) {
  const FlatHashSet<frame_t> frame_ids =
      ConstrainedFrameIds(pose_graph, reconstruction);
  initialized_from_pose_priors_ =
      rig_options_.initialize_from_pose_priors &&
      InitializePositionsFromPosePriors(
          pose_graph, reconstruction, pose_priors, frame_ids);

  if (!initialized_from_pose_priors_) {
    for (const frame_t frame_id : frame_ids) {
      const Frame& frame = reconstruction.Frame(frame_id);
      if (options_.generate_random_positions && options_.optimize_positions) {
        frame_centers_[frame_id] = RandomVector3d(-1, 1);
      } else {
        frame_centers_[frame_id] = frame.RigFromWorld().TgtOriginInSrc();
      }
    }
  }
  VLOG(2) << "Constrained fixed-rig frame positions: " << frame_ids.size();
}

bool FixedRigGlobalPositioner::InitializePositionsFromPosePriors(
    const PoseGraph& pose_graph,
    const Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors,
    const FlatHashSet<frame_t>& constrained_frame_ids) {
  NodeHashMap<image_t, Eigen::Vector3d> image_positions;
  NodeHashMap<frame_t, Eigen::Vector3d> frame_positions;
  for (const PosePrior& prior : pose_priors) {
    if (!prior.HasPosition() ||
        prior.corr_data_id.sensor_id.type != SensorType::CAMERA ||
        !reconstruction.ExistsImage(prior.corr_data_id.id)) {
      continue;
    }
    const image_t image_id = prior.corr_data_id.id;
    const Image& image = reconstruction.Image(image_id);
    image_positions.emplace(image_id, prior.position);
    if (image.IsRefInFrame() && constrained_frame_ids.count(image.FrameId())) {
      frame_positions.emplace(image.FrameId(), prior.position);
    }
  }
  if (frame_positions.size() < constrained_frame_ids.size()) {
    return false;
  }

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (const auto& [pair_id, edge] : pose_graph.ValidEdges()) {
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const auto prior1 = image_positions.find(image_id1);
    const auto prior2 = image_positions.find(image_id2);
    if (prior1 == image_positions.end() || prior2 == image_positions.end()) {
      continue;
    }
    const Eigen::Vector3d prior_direction = prior2->second - prior1->second;
    if (prior_direction.squaredNorm() == 0.0) {
      continue;
    }
    const Eigen::Vector3d world_direction =
        reconstruction.Image(image_id1).CamFromWorld().rotation().inverse() *
        edge.cam2_from_cam1.TgtOriginInSrc();
    covariance += edge.num_matches * world_direction.normalized() *
                  prior_direction.normalized().transpose();
  }
  const Eigen::JacobiSVD<Eigen::Matrix3d> decomposition(
      covariance, Eigen::ComputeFullU | Eigen::ComputeFullV);
  Eigen::Matrix3d sign = Eigen::Matrix3d::Identity();
  sign(2, 2) = (decomposition.matrixU() * decomposition.matrixV().transpose())
                   .determinant();
  const Eigen::Matrix3d world_from_prior =
      decomposition.matrixU() * sign * decomposition.matrixV().transpose();

  Eigen::Vector3d mean = Eigen::Vector3d::Zero();
  for (const frame_t frame_id : constrained_frame_ids) {
    mean += frame_positions.at(frame_id);
  }
  mean /= constrained_frame_ids.size();
  double squared_scale = 0.0;
  for (const frame_t frame_id : constrained_frame_ids) {
    squared_scale += (frame_positions.at(frame_id) - mean).squaredNorm();
  }
  const double scale = std::sqrt(squared_scale / constrained_frame_ids.size());
  for (const frame_t frame_id : constrained_frame_ids) {
    frame_centers_[frame_id] =
        world_from_prior * (frame_positions.at(frame_id) - mean) / scale;
  }
  LOG(INFO) << "Initialized " << constrained_frame_ids.size()
            << " fixed-rig frame positions from pose priors";
  return true;
}

void FixedRigGlobalPositioner::AddPointConstraints(
    Reconstruction& reconstruction) {
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() >=
        static_cast<size_t>(options_.min_num_view_per_track)) {
      AddPointConstraint(point3D_id, reconstruction);
    }
  }
}

void FixedRigGlobalPositioner::AddPointConstraint(
    point3D_t point3D_id, Reconstruction& reconstruction) {
  Point3D& point3D = reconstruction.Point3D(point3D_id);
  if (options_.optimize_points && options_.generate_random_points &&
      !initialized_from_pose_priors_) {
    point3D.xyz = RandomVector3d(-1, 1);
  }

  for (const TrackElement& observation : point3D.track.Elements()) {
    Image& image = reconstruction.Image(observation.image_id);
    if (!image.HasPose()) {
      continue;
    }
    const Eigen::Vector3d cam_ray =
        image.CameraPtr()
            ->CamRayFromImg(image.Point2D(observation.point2D_idx).xy)
            .value();
    const Eigen::Vector3d cam_from_point3D_direction =
        image.CamFromWorld().rotation().inverse() * cam_ray;

    Eigen::Vector3d cam_from_rig_direction = Eigen::Vector3d::Zero();
    if (!image.IsRefInFrame()) {
      const Rigid3d& cam_from_rig =
          reconstruction.Rig(image.FramePtr()->RigId())
              .SensorFromRig(image.CameraPtr()->SensorId());
      THROW_CHECK(!cam_from_rig.translation().hasNaN())
          << "Fixed-rig positioning requires known sensor_from_rig";
      cam_from_rig_direction = image.CamFromWorld().rotation().inverse() *
                               cam_from_rig.translation();
    }

    ceres::LossFunction* loss_function =
        reconstruction.Camera(image.CameraId()).has_prior_focal_length
            ? calibrated_loss_function_.get()
            : uncalibrated_loss_function_.get();
    problem_->AddResidualBlock(
        FixedRigPairwiseDirectionCostFunctor::Create(cam_from_point3D_direction,
                                                     cam_from_rig_direction),
        loss_function,
        point3D.xyz.data(),
        frame_centers_.at(image.FrameId()).data());
  }
}

void FixedRigGlobalPositioner::AddParameterBlockOrdering(
    Reconstruction& reconstruction) {
  options_.solver_options.linear_solver_ordering.reset(
      new ceres::ParameterBlockOrdering);
  ceres::ParameterBlockOrdering& ordering =
      *options_.solver_options.linear_solver_ordering;
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (problem_->HasParameterBlock(point3D.xyz.data())) {
      ordering.AddElementToGroup(reconstruction.Point3D(point3D_id).xyz.data(),
                                 0);
    }
  }
  for (auto& [frame_id, center] : frame_centers_) {
    if (problem_->HasParameterBlock(center.data())) {
      ordering.AddElementToGroup(center.data(), 1);
    }
  }
}

void FixedRigGlobalPositioner::ParameterizeVariables(
    Reconstruction& reconstruction) {
  if (!options_.optimize_positions) {
    for (auto& [frame_id, center] : frame_centers_) {
      if (problem_->HasParameterBlock(center.data())) {
        problem_->SetParameterBlockConstant(center.data());
      }
    }
  }
  if (!options_.optimize_points) {
    for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
      if (problem_->HasParameterBlock(point3D.xyz.data())) {
        problem_->SetParameterBlockConstant(
            reconstruction.Point3D(point3D_id).xyz.data());
      }
    }
  }
  if (options_.optimize_positions) {
    const auto anchor =
        std::min_element(frame_centers_.begin(),
                         frame_centers_.end(),
                         [](const auto& frame1, const auto& frame2) {
                           return frame1.first < frame2.first;
                         });
    if (problem_->HasParameterBlock(anchor->second.data())) {
      problem_->SetParameterBlockConstant(anchor->second.data());
    }
  }
}

void FixedRigGlobalPositioner::ConfigureSolver(
    const Reconstruction& reconstruction) {
#ifdef COLMAP_CUDA_ENABLED
  bool cuda_solver_enabled = false;

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)) && \
    !defined(CERES_NO_CUDA)
  if (options_.use_gpu &&
      reconstruction.NumImages() >=
          static_cast<size_t>(options_.min_num_images_gpu_solver)) {
    cuda_solver_enabled = true;
    options_.solver_options.dense_linear_algebra_library_type = ceres::CUDA;
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested GPU fixed-rig positioning, but Ceres was compiled "
           "without CUDA support. Falling back to CPU.";
  }
#endif

#if (CERES_VERSION_MAJOR >= 3 ||                                \
     (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 3)) && \
    !defined(CERES_NO_CUDSS)
  if (options_.use_gpu &&
      reconstruction.NumImages() >=
          static_cast<size_t>(options_.min_num_images_gpu_solver)) {
    cuda_solver_enabled = true;
    options_.solver_options.sparse_linear_algebra_library_type =
        ceres::CUDA_SPARSE;
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested GPU fixed-rig positioning, but Ceres was compiled "
           "without cuDSS support. Falling back to CPU.";
  }
#endif

  if (cuda_solver_enabled) {
    const std::vector<int> gpu_indices = CSVToVector<int>(options_.gpu_index);
    SetBestCudaDevice(gpu_indices.front());
  }
#else
  if (options_.use_gpu) {
    LOG_FIRST_N(WARNING, 1)
        << "Requested GPU fixed-rig positioning, but COLMAP was compiled "
           "without CUDA support. Falling back to CPU.";
  }
#endif
  options_.solver_options.linear_solver_type = ceres::SPARSE_SCHUR;
}

void FixedRigGlobalPositioner::ConvertBackResults(
    Reconstruction& reconstruction) {
  for (const auto& [frame_id, center] : frame_centers_) {
    Rigid3d& rig_from_world = reconstruction.Frame(frame_id).RigFromWorld();
    rig_from_world.translation() = rig_from_world.rotation() * -center;
  }
}

void FixedRigGlobalPositioner::RefinePoints(
    Reconstruction& reconstruction) const {
  std::vector<point3D_t> point3D_ids;
  point3D_ids.reserve(reconstruction.NumPoints3D());
  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() >=
        static_cast<size_t>(options_.min_num_view_per_track)) {
      point3D_ids.push_back(point3D_id);
    }
  }
  std::sort(point3D_ids.begin(), point3D_ids.end());

  std::vector<Eigen::Vector3d> refined_points(point3D_ids.size());
  const size_t chunk_size =
      (point3D_ids.size() + options_.solver_options.num_threads - 1) /
      options_.solver_options.num_threads;
  const Reconstruction& source = reconstruction;
  ThreadPool thread_pool(options_.solver_options.num_threads);
  for (size_t chunk_begin = 0; chunk_begin < point3D_ids.size();
       chunk_begin += chunk_size) {
    const size_t chunk_end =
        std::min(chunk_begin + chunk_size, point3D_ids.size());
    thread_pool.AddTask([&, chunk_begin, chunk_end]() {
      for (size_t point_index = chunk_begin; point_index < chunk_end;
           ++point_index) {
        const Point3D& point3D = source.Point3D(point3D_ids[point_index]);
        Eigen::Matrix3d normal_matrix = Eigen::Matrix3d::Zero();
        Eigen::Vector3d right_hand_side = Eigen::Vector3d::Zero();
        for (const TrackElement& observation : point3D.track.Elements()) {
          const Image& image = source.Image(observation.image_id);
          const Eigen::Vector3d cam_ray =
              image.CameraPtr()
                  ->CamRayFromImg(image.Point2D(observation.point2D_idx).xy)
                  .value();
          const Eigen::Vector3d world_ray =
              image.CamFromWorld().rotation().inverse() * cam_ray;
          const Eigen::Matrix3d ray_projection =
              Eigen::Matrix3d::Identity() - world_ray * world_ray.transpose();
          normal_matrix += ray_projection;
          right_hand_side += ray_projection * image.ProjectionCenter();
        }
        refined_points[point_index] =
            normal_matrix.ldlt().solve(right_hand_side);
      }
    });
  }
  thread_pool.Wait();

  for (size_t point_index = 0; point_index < point3D_ids.size();
       ++point_index) {
    reconstruction.Point3D(point3D_ids[point_index]).xyz =
        refined_points[point_index];
  }
}

bool FixedRigGlobalPositioner::BuildFrameConstraints(
    const Reconstruction& reconstruction,
    std::vector<FrameConstraint>& constraints) const {
  struct RigObservation {
    frame_t frame_id;
    image_t image_id;
    point2D_t point2D_idx;
    Eigen::Vector3d center;
    Eigen::Vector3d bearing;
  };
  struct RigPoint {
    frame_t frame_id;
    Eigen::Vector3d xyz;
  };

  NodeHashMap<frame_t, std::vector<frame_t>> adjacency;
  adjacency.reserve(frame_centers_.size());
  const double minimum_cosine = std::cos(DegToRad(min_tri_angle_deg_));
  constraints.clear();
  constraints.reserve(reconstruction.NumPoints3D());
  std::vector<RigObservation> observations;
  std::vector<RigPoint> rig_points;

  for (const auto& [point3D_id, point3D] : reconstruction.Points3D()) {
    if (point3D.track.Length() <
        static_cast<size_t>(options_.min_num_view_per_track)) {
      continue;
    }

    observations.clear();
    observations.reserve(point3D.track.Length());
    for (const TrackElement& element : point3D.track.Elements()) {
      const Image& image = reconstruction.Image(element.image_id);
      const Eigen::Vector3d cam_ray =
          image.CameraPtr()
              ->CamRayFromImg(image.Point2D(element.point2D_idx).xy)
              .value();
      Eigen::Vector3d center = Eigen::Vector3d::Zero();
      Eigen::Vector3d bearing = cam_ray;
      if (!image.IsRefInFrame()) {
        const Rigid3d& cam_from_rig =
            reconstruction.Rig(image.FramePtr()->RigId())
                .SensorFromRig(image.CameraPtr()->SensorId());
        center = cam_from_rig.TgtOriginInSrc();
        bearing = cam_from_rig.rotation().inverse() * cam_ray;
      }
      observations.push_back({image.FrameId(),
                              image.ImageId(),
                              element.point2D_idx,
                              center,
                              bearing});
    }
    std::sort(observations.begin(),
              observations.end(),
              [](const RigObservation& observation1,
                 const RigObservation& observation2) {
                return std::tie(observation1.frame_id,
                                observation1.image_id,
                                observation1.point2D_idx) <
                       std::tie(observation2.frame_id,
                                observation2.image_id,
                                observation2.point2D_idx);
              });

    rig_points.clear();
    for (size_t group_begin = 0; group_begin < observations.size();) {
      size_t group_end = group_begin + 1;
      while (group_end < observations.size() &&
             observations[group_end].frame_id ==
                 observations[group_begin].frame_id) {
        ++group_end;
      }
      if (group_end - group_begin < 2) {
        group_begin = group_end;
        continue;
      }

      bool has_distinct_images = false;
      double smallest_cosine = 1.0;
      for (size_t index1 = group_begin; index1 < group_end; ++index1) {
        for (size_t index2 = index1 + 1; index2 < group_end; ++index2) {
          has_distinct_images |=
              observations[index1].image_id != observations[index2].image_id;
          smallest_cosine = std::min(
              smallest_cosine,
              observations[index1].bearing.dot(observations[index2].bearing));
        }
      }
      if (!has_distinct_images || smallest_cosine > minimum_cosine) {
        group_begin = group_end;
        continue;
      }

      Eigen::Matrix3d normal_matrix = Eigen::Matrix3d::Zero();
      Eigen::Vector3d right_hand_side = Eigen::Vector3d::Zero();
      for (size_t index = group_begin; index < group_end; ++index) {
        const Eigen::Vector3d& bearing = observations[index].bearing;
        const Eigen::Matrix3d ray_projection =
            Eigen::Matrix3d::Identity() - bearing * bearing.transpose();
        normal_matrix += ray_projection;
        right_hand_side += ray_projection * observations[index].center;
      }
      const Eigen::Vector3d xyz = normal_matrix.ldlt().solve(right_hand_side);
      bool has_positive_depth = true;
      for (size_t index = group_begin; index < group_end; ++index) {
        if (observations[index].bearing.dot(xyz - observations[index].center) <=
            0.0) {
          has_positive_depth = false;
          break;
        }
      }
      if (has_positive_depth) {
        rig_points.push_back({observations[group_begin].frame_id, xyz});
      }
      group_begin = group_end;
    }

    for (size_t index = 1; index < rig_points.size(); ++index) {
      const RigPoint& point1 = rig_points[index - 1];
      const RigPoint& point2 = rig_points[index];
      const Eigen::Vector3d center2_from_center1 =
          reconstruction.Frame(point1.frame_id)
                  .RigFromWorld()
                  .rotation()
                  .inverse() *
              point1.xyz -
          reconstruction.Frame(point2.frame_id)
                  .RigFromWorld()
                  .rotation()
                  .inverse() *
              point2.xyz;
      constraints.push_back(
          {point1.frame_id, point2.frame_id, center2_from_center1, 1.0});
    }
  }

  std::sort(constraints.begin(),
            constraints.end(),
            [](const FrameConstraint& constraint1,
               const FrameConstraint& constraint2) {
              return std::make_tuple(constraint1.frame_id1,
                                     constraint1.frame_id2,
                                     constraint1.center2_from_center1.x(),
                                     constraint1.center2_from_center1.y(),
                                     constraint1.center2_from_center1.z()) <
                     std::make_tuple(constraint2.frame_id1,
                                     constraint2.frame_id2,
                                     constraint2.center2_from_center1.x(),
                                     constraint2.center2_from_center1.y(),
                                     constraint2.center2_from_center1.z());
            });
  std::vector<FrameConstraint> aggregated_constraints;
  for (size_t group_begin = 0; group_begin < constraints.size();) {
    size_t group_end = group_begin + 1;
    while (group_end < constraints.size() &&
           constraints[group_end].frame_id1 ==
               constraints[group_begin].frame_id1 &&
           constraints[group_end].frame_id2 ==
               constraints[group_begin].frame_id2) {
      ++group_end;
    }

    Eigen::Vector3d median;
    std::vector<double> coordinates(group_end - group_begin);
    for (int coordinate = 0; coordinate < 3; ++coordinate) {
      for (size_t index = group_begin; index < group_end; ++index) {
        coordinates[index - group_begin] =
            constraints[index].center2_from_center1(coordinate);
      }
      median(coordinate) = Median(coordinates);
    }
    std::vector<std::pair<double, size_t>> distances;
    distances.reserve(group_end - group_begin);
    for (size_t index = group_begin; index < group_end; ++index) {
      distances.emplace_back(
          (constraints[index].center2_from_center1 - median).squaredNorm(),
          index);
    }
    std::sort(distances.begin(), distances.end());
    const size_t num_inliers = (9 * distances.size() + 9) / 10;
    Eigen::Vector3d mean = Eigen::Vector3d::Zero();
    for (size_t index = 0; index < num_inliers; ++index) {
      mean += constraints[distances[index].second].center2_from_center1;
    }
    mean /= num_inliers;
    aggregated_constraints.push_back(
        {constraints[group_begin].frame_id1,
         constraints[group_begin].frame_id2,
         mean,
         std::sqrt(static_cast<double>(num_inliers))});
    group_begin = group_end;
  }
  constraints = std::move(aggregated_constraints);
  for (const FrameConstraint& constraint : constraints) {
    adjacency[constraint.frame_id1].push_back(constraint.frame_id2);
    adjacency[constraint.frame_id2].push_back(constraint.frame_id1);
  }
  if (adjacency.size() != frame_centers_.size()) {
    std::vector<frame_t> uncovered_frame_ids;
    uncovered_frame_ids.reserve(frame_centers_.size() - adjacency.size());
    for (const auto& [frame_id, _] : frame_centers_) {
      if (adjacency.count(frame_id) == 0) {
        uncovered_frame_ids.push_back(frame_id);
      }
    }
    std::sort(uncovered_frame_ids.begin(), uncovered_frame_ids.end());
    LOG(ERROR) << "Calibrated-rig frame constraints leave "
               << uncovered_frame_ids.size() << " of " << frame_centers_.size()
               << " frames uncovered: " << VectorToCSV(uncovered_frame_ids);
    return false;
  }
  return true;
}

bool FixedRigGlobalPositioner::SolveFramePositions(
    Reconstruction& reconstruction,
    const std::vector<FrameConstraint>& constraints) {
  NodeHashMap<frame_t, std::vector<frame_t>> adjacency;
  adjacency.reserve(frame_centers_.size());
  for (const FrameConstraint& constraint : constraints) {
    problem_->AddResidualBlock(
        FixedRigFramePositionCostFunctor::Create(
            constraint.center2_from_center1, constraint.weight),
        loss_function_.get(),
        frame_centers_.at(constraint.frame_id1).data(),
        frame_centers_.at(constraint.frame_id2).data());
    adjacency[constraint.frame_id1].push_back(constraint.frame_id2);
    adjacency[constraint.frame_id2].push_back(constraint.frame_id1);
  }

  std::vector<frame_t> frame_ids;
  frame_ids.reserve(frame_centers_.size());
  for (const auto& [frame_id, _] : frame_centers_) {
    frame_ids.push_back(frame_id);
  }
  std::sort(frame_ids.begin(), frame_ids.end());
  FlatHashSet<frame_t> visited;
  size_t component_count = 0;
  for (const frame_t anchor_frame_id : frame_ids) {
    if (visited.count(anchor_frame_id) > 0) {
      continue;
    }
    ++component_count;
    problem_->SetParameterBlockConstant(
        frame_centers_.at(anchor_frame_id).data());
    std::queue<frame_t> queue;
    queue.push(anchor_frame_id);
    while (!queue.empty()) {
      const frame_t frame_id = queue.front();
      queue.pop();
      if (!visited.insert(frame_id).second) {
        continue;
      }
      for (const frame_t neighbor : adjacency.at(frame_id)) {
        queue.push(neighbor);
      }
    }
  }
  LOG(INFO) << "Solving calibrated-rig frame constraints in " << component_count
            << " connected component(s)";

  ceres::Solver::Options solver_options = options_.solver_options;
  solver_options.linear_solver_type = ceres::SPARSE_NORMAL_CHOLESKY;
  solver_options.linear_solver_ordering.reset();
  solver_options.minimizer_progress_to_stdout = VLOG_IS_ON(2);
  ceres::Solver::Summary summary;
  ceres::Solve(solver_options, problem_.get(), &summary);
  LOG(INFO) << (VLOG_IS_ON(2) ? summary.FullReport() : summary.BriefReport());

  ConvertBackResults(reconstruction);
  RefinePoints(reconstruction);
  return summary.IsSolutionUsable();
}

bool RunFixedRigGlobalPositioning(
    const GlobalPositionerOptions& options,
    const FixedRigGlobalPositionerOptions& rig_options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors,
    double min_tri_angle_deg) {
  FixedRigGlobalPositioner positioner(options, rig_options, min_tri_angle_deg);
  return positioner.Solve(pose_graph, reconstruction, pose_priors);
}

}  // namespace colmap
