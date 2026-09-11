#include "colmap/sfm/global_mapper.h"

#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/estimators/rotation_averaging.h"
#include "colmap/scene/projection.h"
#include "colmap/sfm/incremental_mapper.h"
#include "colmap/sfm/observation_manager.h"
#include "colmap/sfm/valid_correspondences.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"
#include "colmap/util/timer.h"

#include <algorithm>
#include <limits>
#include <numeric>

namespace colmap {
namespace {

bool RunBundleAdjustment(const BundleAdjustmentOptions& options,
                         Reconstruction& reconstruction) {
  if (reconstruction.NumImages() == 0) {
    LOG(ERROR) << "Cannot run bundle adjustment: no registered images";
    return false;
  }
  if (reconstruction.NumPoints3D() == 0) {
    LOG(ERROR) << "Cannot run bundle adjustment: no 3D points to optimize";
    return false;
  }

  BundleAdjustmentConfig ba_config;
  for (const auto& [image_id, image] : reconstruction.Images()) {
    if (image.HasPose()) {
      ba_config.AddImage(image_id);
    }
  }
  ba_config.FixGauge(BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD);

  auto ba = CreateDefaultBundleAdjuster(options, ba_config, reconstruction);

  return ba->Solve()->IsSolutionUsable();
}

void CapBundleAdjustmentIterations(BundleAdjustmentOptions& options,
                                   int max_num_iterations) {
  if (options.ceres) {
    options.ceres->solver_options.max_num_iterations = std::min(
        options.ceres->solver_options.max_num_iterations, max_num_iterations);
  }
  if (options.caspar) {
    options.caspar->solver_iter_max =
        std::min(options.caspar->solver_iter_max, max_num_iterations);
  }
}

}  // namespace

bool GlobalMapperOptions::Check() const {
  CHECK_OPTION_GE(retriangulation_max_num_refinements, 0);
  CHECK_OPTION_GT(retriangulation_ba_max_num_iterations, 0);
  return true;
}

RotationEstimatorOptions GlobalMapperOptions::RotationAveraging() const {
  RotationEstimatorOptions opts = rotation_averaging;
  opts.refine_sensor_from_rig = refine_sensor_from_rig;
  if (random_seed >= 0) {
    opts.random_seed = random_seed;
  }
  return opts;
}

GlobalPositionerOptions GlobalMapperOptions::GlobalPositioning() const {
  GlobalPositionerOptions opts = global_positioning;
  opts.refine_sensor_from_rig = refine_sensor_from_rig;
  opts.solver_options.num_threads = num_threads;
  if (random_seed >= 0) {
    opts.random_seed = random_seed;
    opts.use_parameter_block_ordering = false;
  }
  return opts;
}

BundleAdjustmentOptions GlobalMapperOptions::BundleAdjustment() const {
  BundleAdjustmentOptions opts = bundle_adjustment;
  opts.refine_sensor_from_rig = refine_sensor_from_rig;
  if (opts.ceres) {
    opts.ceres->solver_options.num_threads = num_threads;
    opts.ceres->gpu_index = ba_gpu_index;
  }
  if (opts.caspar) {
    opts.caspar->gpu_index = ba_gpu_index;
  }
  return opts;
}

IncrementalTriangulator::Options GlobalMapperOptions::Retriangulation() const {
  IncrementalTriangulator::Options opts = retriangulation;
  if (random_seed >= 0) {
    opts.random_seed = random_seed;
  }
  return opts;
}

GlobalMapperOptions GlobalMapperStrategy::Configure(
    const GlobalMapperOptions& options) const {
  THROW_CHECK(options.bundle_adjustment.backend !=
              BundleAdjustmentBackend::CASPAR_RIG_SCHUR)
      << "Caspar rig-Schur is only supported by fixed-rig global mapping";
  return options;
}

void GlobalMapperStrategy::PrepareRotationAveraging(
    PoseGraph&, const Reconstruction&, const RotationEstimatorOptions&) const {}

bool GlobalMapperStrategy::RunPositioning(
    const GlobalPositionerOptions& options,
    const PoseGraph& pose_graph,
    Reconstruction& reconstruction,
    const std::vector<PosePrior>&,
    double) const {
  return RunGlobalPositioning(options, pose_graph, reconstruction);
}

std::shared_ptr<const GlobalMapperStrategy> CreateGlobalMapperStrategy() {
  return std::make_shared<GlobalMapperStrategy>();
}

GlobalMapper::GlobalMapper(std::shared_ptr<const DatabaseCache> database_cache)
    : GlobalMapper(std::move(database_cache), CreateGlobalMapperStrategy()) {}

GlobalMapper::GlobalMapper(std::shared_ptr<const DatabaseCache> database_cache,
                           std::shared_ptr<const GlobalMapperStrategy> strategy)
    : database_cache_(std::move(THROW_CHECK_NOTNULL(database_cache))),
      strategy_(std::move(THROW_CHECK_NOTNULL(strategy))) {}

void GlobalMapper::BeginReconstruction(
    const std::shared_ptr<class Reconstruction>& reconstruction) {
  THROW_CHECK_NOTNULL(reconstruction);
  reconstruction_ = reconstruction;
  reconstruction_->Load(*database_cache_);
  pose_graph_ = std::make_shared<class PoseGraph>();
  pose_graph_->Load(*database_cache_->CorrespondenceGraph());
}

std::shared_ptr<Reconstruction> GlobalMapper::Reconstruction() const {
  return reconstruction_;
}

bool GlobalMapper::RotationAveraging(const RotationEstimatorOptions& options) {
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(pose_graph_);

  if (pose_graph_->Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  strategy_->PrepareRotationAveraging(*pose_graph_, *reconstruction_, options);

  // Read pose priors from the database cache.
  const std::vector<PosePrior>& pose_priors = database_cache_->PosePriors();

  // First pass: solve rotation averaging on all frames, then filter outlier
  // pairs by rotation error and de-register frames outside the largest
  // connected component.
  RotationEstimatorOptions custom_options = options;
  custom_options.filter_unregistered = false;
  if (!RunRotationAveraging(
          custom_options, *pose_graph_, *reconstruction_, pose_priors)) {
    return false;
  }

  // Second pass: re-solve on registered frames only to refine rotations
  // after outlier removal.
  custom_options.filter_unregistered = true;
  if (!RunRotationAveraging(
          custom_options, *pose_graph_, *reconstruction_, pose_priors)) {
    return false;
  }

  VLOG(1) << reconstruction_->NumRegImages() << " / "
          << reconstruction_->NumImages()
          << " images are within the connected component.";

  return true;
}

void GlobalMapper::EstablishTracks(const GlobalMapperOptions& options) {
  THROW_CHECK_EQ(reconstruction_->NumPoints3D(), 0);

  auto corr_graph = database_cache_->CorrespondenceGraph();
  std::vector<image_t> image_ids = reconstruction_->RegImageIds();
  std::sort(image_ids.begin(), image_ids.end());

  FlatHashMap<image_t, size_t> image_indices;
  image_indices.reserve(image_ids.size());
  for (size_t image_index = 0; image_index < image_ids.size(); ++image_index) {
    image_indices.emplace(image_ids[image_index], image_index);
  }

  std::vector<TrackElement> observations;
  std::vector<size_t> grouped_observation_ids;
  std::vector<size_t> component_offsets;
  std::vector<size_t> component_roots;
  {
    std::vector<size_t> parents;
    {
      const size_t invalid_observation_id = std::numeric_limits<size_t>::max();
      std::vector<std::vector<size_t>> observation_ids;
      observation_ids.reserve(image_ids.size());
      for (const image_t image_id : image_ids) {
        observation_ids.emplace_back(
            reconstruction_->Image(image_id).NumPoints2D(),
            invalid_observation_id);
      }

      ForEachValidCorrespondence(
          *corr_graph,
          *pose_graph_,
          *reconstruction_,
          image_ids,
          image_indices,
          [&](size_t first, point2D_t point1, size_t second, point2D_t point2) {
            observation_ids[first][point1] = 0;
            observation_ids[second][point2] = 0;
          });

      for (size_t image_index = 0; image_index < image_ids.size();
           ++image_index) {
        auto& image_observation_ids = observation_ids[image_index];
        for (size_t point2D_idx = 0; point2D_idx < image_observation_ids.size();
             ++point2D_idx) {
          if (image_observation_ids[point2D_idx] == invalid_observation_id) {
            continue;
          }
          image_observation_ids[point2D_idx] = observations.size();
          observations.emplace_back(image_ids[image_index],
                                    static_cast<point2D_t>(point2D_idx));
        }
      }

      parents.resize(observations.size());
      std::iota(parents.begin(), parents.end(), 0);
      std::vector<size_t> component_sizes(observations.size(), 1);
      const auto find_root = [&parents](size_t observation_id) {
        while (parents[observation_id] != observation_id) {
          parents[observation_id] = parents[parents[observation_id]];
          observation_id = parents[observation_id];
        }
        return observation_id;
      };

      ForEachValidCorrespondence(
          *corr_graph,
          *pose_graph_,
          *reconstruction_,
          image_ids,
          image_indices,
          [&](size_t first, point2D_t point1, size_t second, point2D_t point2) {
            size_t root1 = find_root(observation_ids[first][point1]);
            size_t root2 = find_root(observation_ids[second][point2]);
            if (root1 == root2) return;
            if (component_sizes[root1] < component_sizes[root2] ||
                (component_sizes[root1] == component_sizes[root2] &&
                 root2 < root1)) {
              std::swap(root1, root2);
            }
            parents[root2] = root1;
            component_sizes[root1] += component_sizes[root2];
          });

      for (size_t observation_id = 0; observation_id < parents.size();
           ++observation_id) {
        parents[observation_id] = find_root(observation_id);
      }
    }

    component_offsets.assign(observations.size() + 1, 0);
    for (const size_t root : parents) {
      ++component_offsets[root + 1];
    }
    for (size_t root = 0; root < parents.size(); ++root) {
      if (component_offsets[root + 1] != 0) {
        component_roots.push_back(root);
      }
    }
    std::partial_sum(component_offsets.begin(),
                     component_offsets.end(),
                     component_offsets.begin());

    grouped_observation_ids.resize(observations.size());
    for (size_t observation_id = observations.size(); observation_id > 0;
         --observation_id) {
      const size_t id = observation_id - 1;
      const size_t root = parents[id];
      grouped_observation_ids[--component_offsets[root + 1]] = id;
    }
  }

  LOG(INFO) << "Established " << component_roots.size() << " tracks from "
            << observations.size() << " observations";

  struct TrackCandidate {
    size_t begin;
    size_t end;
    size_t canonical_observation_id;
  };
  std::vector<TrackCandidate> candidates;
  candidates.reserve(component_roots.size());
  size_t discarded_counter = 0;
  const double squared_consistency_threshold =
      options.track_intra_image_consistency_threshold *
      options.track_intra_image_consistency_threshold;

  for (size_t component_index = 0; component_index < component_roots.size();
       ++component_index) {
    const size_t begin =
        component_offsets[component_roots[component_index] + 1];
    const size_t end =
        component_index + 1 < component_roots.size()
            ? component_offsets[component_roots[component_index + 1] + 1]
            : observations.size();
    bool is_consistent = true;
    size_t num_images = 0;
    size_t image_begin = begin;
    while (image_begin < end) {
      const image_t image_id =
          observations[grouped_observation_ids[image_begin]].image_id;
      size_t image_end = image_begin + 1;
      while (image_end < end &&
             observations[grouped_observation_ids[image_end]].image_id ==
                 image_id) {
        ++image_end;
      }
      ++num_images;

      const auto& image = reconstruction_->Image(image_id);
      for (size_t observation_index = image_begin + 1;
           observation_index < image_end;
           ++observation_index) {
        const auto& observation =
            observations[grouped_observation_ids[observation_index]];
        const Eigen::Vector2d& xy = image.Point2D(observation.point2D_idx).xy;
        for (size_t other_index = image_begin; other_index < observation_index;
             ++other_index) {
          const auto& other =
              observations[grouped_observation_ids[other_index]];
          if ((image.Point2D(other.point2D_idx).xy - xy).squaredNorm() >
              squared_consistency_threshold) {
            is_consistent = false;
            break;
          }
        }
        if (!is_consistent) {
          break;
        }
      }
      if (!is_consistent) {
        break;
      }
      image_begin = image_end;
    }

    if (!is_consistent) {
      ++discarded_counter;
      continue;
    }
    if (num_images <
        static_cast<size_t>(options.track_min_num_views_per_track)) {
      continue;
    }
    candidates.push_back({begin, end, grouped_observation_ids[begin]});
  }

  LOG(INFO) << "Kept " << candidates.size() << " tracks, discarded "
            << discarded_counter << " due to inconsistency";

  std::sort(candidates.begin(),
            candidates.end(),
            [](const auto& candidate1, const auto& candidate2) {
              const size_t length1 = candidate1.end - candidate1.begin;
              const size_t length2 = candidate2.end - candidate2.begin;
              return length1 != length2
                         ? length1 > length2
                         : candidate1.canonical_observation_id <
                               candidate2.canonical_observation_id;
            });

  FlatHashMap<image_t, size_t> tracks_per_image;
  size_t images_left = image_ids.size();
  const size_t max_num_tracks =
      static_cast<size_t>(options.keep_max_num_tracks);
  point3D_t next_point3D_id = 0;
  for (const auto& candidate : candidates) {
    // Stop once the global track budget is exhausted. As tracks are sorted by
    // decreasing length, this keeps the longest tracks and bounds memory usage.
    if (reconstruction_->NumPoints3D() >= max_num_tracks) break;

    // Check if any image in this track still needs more observations.
    bool should_add = false;
    for (size_t index = candidate.begin; index < candidate.end; ++index) {
      const auto& observation = observations[grouped_observation_ids[index]];
      if (tracks_per_image[observation.image_id] <=
          static_cast<size_t>(options.track_required_tracks_per_view)) {
        should_add = true;
        break;
      }
    }
    if (!should_add) continue;

    Point3D point3D;
    point3D.track.Reserve(candidate.end - candidate.begin);
    // Update image counts.
    for (size_t index = candidate.begin; index < candidate.end; ++index) {
      const auto& observation = observations[grouped_observation_ids[index]];
      point3D.track.AddElement(observation);
      auto& count = tracks_per_image[observation.image_id];
      if (count == static_cast<size_t>(options.track_required_tracks_per_view))
        --images_left;
      ++count;
    }

    // Add track after updating counts so we can move.
    reconstruction_->AddPoint3D(next_point3D_id++, std::move(point3D));

    if (images_left == 0) break;
  }

  LOG(INFO) << "Before filtering: " << candidates.size()
            << ", after filtering: " << reconstruction_->NumPoints3D();
}

bool GlobalMapper::GlobalPositioning(const GlobalPositionerOptions& options,
                                     double max_angular_reproj_error_deg,
                                     double max_normalized_reproj_error,
                                     double min_tri_angle_deg) {
  if (!strategy_->RunPositioning(options,
                                 *pose_graph_,
                                 *reconstruction_,
                                 database_cache_->PosePriors(),
                                 min_tri_angle_deg)) {
    return false;
  }

  // Filter tracks based on the estimation
  ObservationManager obs_manager(*reconstruction_);
  const FlatHashSet<point3D_t> point3D_ids = reconstruction_->Point3DIds();

  // Apply the strict threshold to calibrated cameras and a relaxed threshold
  // to cameras whose focal length was estimated.
  const double max_angular_error_rad = DegToRad(max_angular_reproj_error_deg);
  std::vector<TrackElement> observations_to_delete;
  for (const point3D_t point3D_id : point3D_ids) {
    if (!reconstruction_->ExistsPoint3D(point3D_id)) {
      continue;
    }
    const auto& point3D = reconstruction_->Point3D(point3D_id);
    if (point3D.track.Length() < 2) {
      obs_manager.DeletePoint3D(point3D_id);
      continue;
    }

    observations_to_delete.clear();
    for (const auto& track_el : point3D.track.Elements()) {
      const auto& image = reconstruction_->Image(track_el.image_id);
      const auto& camera = *image.CameraPtr();
      const auto& point2D = image.Point2D(track_el.point2D_idx);
      const double error = CalculateAngularReprojectionError(
          point2D.xy, point3D.xyz, image.CamFromWorld(), camera);
      const bool exceeds_threshold =
          camera.has_prior_focal_length
              ? error > max_angular_error_rad
              : RadToDeg(error) > 2.0 * max_angular_reproj_error_deg;
      if (exceeds_threshold) {
        observations_to_delete.push_back(track_el);
      }
    }

    if (observations_to_delete.size() >= point3D.track.Length() - 1) {
      obs_manager.DeletePoint3D(point3D_id);
    } else {
      for (const TrackElement& observation : observations_to_delete) {
        obs_manager.DeleteObservation(observation.image_id,
                                      observation.point2D_idx);
      }
    }
  }

  // Filter tracks based on triangulation angle and reprojection error
  obs_manager.FilterPoints3DWithSmallTriangulationAngle(min_tri_angle_deg,
                                                        point3D_ids);
  // Set the threshold to be larger to avoid removing too many tracks
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      10 * max_normalized_reproj_error,
      point3D_ids,
      ReprojectionErrorType::NORMALIZED);

  // Normalize the structure for numerical stability.
  // TODO: Skip normalization when position priors are used (similar to
  // incremental mapper's !use_prior_position condition).
  reconstruction_->Normalize();

  return true;
}

bool GlobalMapper::IterativeBundleAdjustment(
    const BundleAdjustmentOptions& options,
    double max_normalized_reproj_error,
    double min_tri_angle_deg,
    int num_iterations,
    bool skip_fixed_rotation_stage,
    bool skip_joint_optimization_stage,
    const std::function<bool()>& on_progress) {
  bool filtered_at_final_threshold = false;
  for (int ite = 0; ite < num_iterations; ite++) {
    filtered_at_final_threshold = false;

    // Optional fixed-rotation stage: optimize positions only
    if (!skip_fixed_rotation_stage) {
      BundleAdjustmentOptions opts_position_only = options;
      opts_position_only.constant_rig_from_world_rotation = true;
      // Caspar's pose node is a single retracted Pose3 (rotation+
      // translation together) with no mechanism to hold rotation
      // constant while translation is free -- constant_rig_from_world_
      // rotation is silently ignored when backend == CASPAR.
      if (opts_position_only.backend == BundleAdjustmentBackend::CASPAR) {
        opts_position_only.backend = BundleAdjustmentBackend::CERES;
      }
      if (!RunBundleAdjustment(opts_position_only, *reconstruction_)) {
        return false;
      }
      LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
                << num_iterations << ", fixed-rotation stage finished";
    }

    // Joint optimization stage: default BA
    if (!skip_joint_optimization_stage) {
      if (!RunBundleAdjustment(options, *reconstruction_)) {
        return false;
      }
    }
    LOG(INFO) << "Global bundle adjustment iteration " << ite + 1 << " / "
              << num_iterations << " finished";

    // Normalize the structure for numerical stability.
    // TODO: Skip normalization when position priors are used (similar to
    // incremental mapper's !use_prior_position condition).
    reconstruction_->Normalize();

    // Report progress for this refinement iteration and stop early if
    // requested. The filter passes above leave point3D.error in normalized
    // units, so recompute it in pixels first to keep intermediate
    // visualizations consistent with the final reconstruction.
    if (on_progress) {
      reconstruction_->UpdatePoint3DErrors();
      if (on_progress()) {
        break;
      }
    }

    // Filter tracks based on the estimation
    // For the filtering, in each round, the criteria for outlier is
    // tightened. If only few tracks are changed, no need to start bundle
    // adjustment right away. Instead, use a more strict criteria to filter
    LOG(INFO) << "Filtering tracks by reprojection ...";

    ObservationManager obs_manager(*reconstruction_);
    bool status = true;
    size_t filtered_num = 0;
    const FlatHashSet<point3D_t> point3D_ids = reconstruction_->Point3DIds();
    while (status && ite < num_iterations) {
      double scaling = std::max(3 - ite, 1);
      filtered_num += obs_manager.FilterPoints3DWithLargeReprojectionError(
          scaling * max_normalized_reproj_error,
          point3D_ids,
          ReprojectionErrorType::NORMALIZED);
      filtered_at_final_threshold = scaling == 1.0;

      if (filtered_num > 1e-3 * reconstruction_->NumPoints3D()) {
        status = false;
      } else {
        ite++;
      }
    }
    if (status) {
      LOG(INFO) << "fewer than 0.1% tracks are filtered, stop the iteration.";
      break;
    }
  }

  {
    ObservationManager obs_manager(*reconstruction_);
    const FlatHashSet<point3D_t> point3D_ids = reconstruction_->Point3DIds();
    if (!filtered_at_final_threshold) {
      LOG(INFO) << "Filtering tracks by reprojection ...";
      obs_manager.FilterPoints3DWithLargeReprojectionError(
          max_normalized_reproj_error,
          point3D_ids,
          ReprojectionErrorType::NORMALIZED);
    }
    obs_manager.FilterPoints3DWithSmallTriangulationAngle(min_tri_angle_deg,
                                                          point3D_ids);
  }

  return true;
}

bool GlobalMapper::IterativeRetriangulateAndRefine(
    const IncrementalTriangulator::Options& options,
    const BundleAdjustmentOptions& ba_options,
    double max_normalized_reproj_error,
    double min_tri_angle_deg,
    int max_num_refinements,
    int ba_max_num_iterations) {
  // Delete all existing 3D points and re-establish 2D-3D correspondences.
  reconstruction_->DeleteAllPoints2DAndPoints3D();

  // Initialize mapper.
  IncrementalMapper mapper(database_cache_);
  mapper.BeginReconstruction(reconstruction_);

  // Triangulate all registered images.
  for (const auto image_id : reconstruction_->RegImageIds()) {
    mapper.TriangulateImage(options, image_id);
  }

  // Set up bundle adjustment options for colmap's incremental mapper.
  BundleAdjustmentOptions custom_ba_options = ba_options;
  custom_ba_options.print_summary = false;
  if (custom_ba_options.ceres && ba_options.ceres) {
    custom_ba_options.ceres->solver_options.num_threads =
        ba_options.ceres->solver_options.num_threads;
    custom_ba_options.ceres->solver_options.max_linear_solver_iterations = 100;
  }
  CapBundleAdjustmentIterations(custom_ba_options, ba_max_num_iterations);

  // Iterative global refinement.
  IncrementalMapper::Options mapper_options;
  mapper_options.random_seed = options.random_seed;
  mapper.IterativeGlobalRefinement(max_num_refinements,
                                   /*max_refinement_change=*/0.0005,
                                   mapper_options,
                                   custom_ba_options,
                                   options,
                                   /*normalize_reconstruction=*/true);

  mapper.EndReconstruction(/*discard=*/false);

  // Final filtering and bundle adjustment.
  ObservationManager obs_manager(*reconstruction_);
  obs_manager.FilterPoints3DWithLargeReprojectionError(
      max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::NORMALIZED);

  BundleAdjustmentOptions final_ba_options = ba_options;
  CapBundleAdjustmentIterations(final_ba_options, ba_max_num_iterations);
  if (!RunBundleAdjustment(final_ba_options, *reconstruction_)) {
    return false;
  }

  // Normalize the structure for numerical stability.
  // TODO: Skip normalization when position priors are used (similar to
  // incremental mapper's !use_prior_position condition).
  reconstruction_->Normalize();

  obs_manager.FilterPoints3DWithLargeReprojectionError(
      max_normalized_reproj_error,
      reconstruction_->Point3DIds(),
      ReprojectionErrorType::NORMALIZED);
  obs_manager.FilterPoints3DWithSmallTriangulationAngle(
      min_tri_angle_deg, reconstruction_->Point3DIds());

  return true;
}

bool GlobalMapper::Solve(const GlobalMapperOptions& input_options,
                         const std::function<bool()>& on_progress) {
  const GlobalMapperOptions options = strategy_->Configure(input_options);
  THROW_CHECK(options.Check());
  THROW_CHECK_NOTNULL(reconstruction_);
  THROW_CHECK_NOTNULL(pose_graph_);

  if (pose_graph_->Empty()) {
    LOG(ERROR) << "Cannot continue with empty pose graph";
    return false;
  }

  // Reports the current reconstruction and returns whether a stop was
  // requested. Point errors are recomputed in pixels before reporting because
  // the preceding filter passes leave point3D.error in normalized units, which
  // would otherwise make intermediate visualizations inconsistent with the
  // final reconstruction.
  const auto report_and_check_stop = [&]() {
    if (!on_progress) {
      return false;
    }
    reconstruction_->UpdatePoint3DErrors();
    return on_progress();
  };

  // Run rotation averaging
  if (!options.skip_rotation_averaging) {
    LOG_HEADING1("Running rotation averaging");
    Timer run_timer;
    run_timer.Start();
    if (!RotationAveraging(options.RotationAveraging())) {
      return false;
    }
    LOG(INFO) << "Rotation averaging done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Track establishment and selection
  if (!options.skip_track_establishment) {
    LOG_HEADING1("Running track establishment");
    Timer run_timer;
    run_timer.Start();
    EstablishTracks(options);
    LOG(INFO) << "Track establishment done in " << run_timer.ElapsedSeconds()
              << " seconds";
  }

  // Global positioning
  if (!options.skip_global_positioning) {
    LOG_HEADING1("Running global positioning");
    Timer run_timer;
    run_timer.Start();
    if (!GlobalPositioning(options.GlobalPositioning(),
                           options.max_angular_reproj_error_deg,
                           options.max_normalized_reproj_error,
                           options.min_tri_angle_deg)) {
      return false;
    }
    LOG(INFO) << "Global positioning done in " << run_timer.ElapsedSeconds()
              << " seconds";

    // Report the first 3D view after global positioning and stop early if
    // requested.
    if (report_and_check_stop()) {
      return true;
    }
  }

  // Bundle adjustment
  if (!options.skip_bundle_adjustment) {
    LOG_HEADING1("Running iterative bundle adjustment");
    Timer run_timer;
    run_timer.Start();
    if (!IterativeBundleAdjustment(options.BundleAdjustment(),
                                   options.max_normalized_reproj_error,
                                   options.min_tri_angle_deg,
                                   options.ba_num_iterations,
                                   options.ba_skip_fixed_rotation_stage,
                                   options.ba_skip_joint_optimization_stage,
                                   on_progress)) {
      return false;
    }
    LOG(INFO) << "Iterative bundle adjustment done in "
              << run_timer.ElapsedSeconds() << " seconds";
  }

  // Retriangulation
  if (!options.skip_retriangulation) {
    LOG_HEADING1("Running iterative retriangulation and refinement");
    Timer run_timer;
    run_timer.Start();
    if (!IterativeRetriangulateAndRefine(
            options.Retriangulation(),
            options.BundleAdjustment(),
            options.max_normalized_reproj_error,
            options.min_tri_angle_deg,
            options.retriangulation_max_num_refinements,
            options.retriangulation_ba_max_num_iterations)) {
      return false;
    }
    LOG(INFO) << "Iterative retriangulation and refinement done in "
              << run_timer.ElapsedSeconds() << " seconds";

    // Report the result after retriangulation and stop early if requested.
    if (report_and_check_stop()) {
      return true;
    }
  }

  // Filter passes here use NORMALIZED/ANGULAR error, so point3D.error is
  // left in non-pixel units. Recompute in pixels for consistent reporting
  // in model_analyzer.
  reconstruction_->UpdatePoint3DErrors();

  return true;
}

}  // namespace colmap
