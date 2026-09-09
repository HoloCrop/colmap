#include "colmap/estimators/row_bundle_adjustment_caspar.h"

#include "colmap/estimators/bundle_adjustment_arrays_caspar.h"
#include "colmap/estimators/caspar/caspar_model_adapter.h"
#include "colmap/geometry/sim3.h"
#include "colmap/util/cuda.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <utility>

#include <Eigen/Core>

namespace colmap {
namespace {

using Clock = std::chrono::steady_clock;
using PackedSensorCalibration = Eigen::Matrix<StorageType, 11, 1>;
using InputCalibration = Eigen::Matrix<float, 4, 1>;
using InputSqrtInformation = Eigen::Matrix<float, 3, 3, Eigen::RowMajor>;
using PackedSqrtInformation = Eigen::Matrix<StorageType, 3, 3>;

double ElapsedSeconds(const Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

struct ObservationFactorData {
  std::vector<unsigned int> pose_indices;
  std::vector<unsigned int> point_indices;
  std::vector<unsigned int> sensor_indices;
  std::vector<StorageType> pixels;
};

struct PriorFactorData {
  std::vector<unsigned int> pose_indices;
  std::vector<StorageType> positions;
  std::vector<StorageType> sqrt_information;
};

struct NormalizedPriors {
  std::vector<StorageType> positions;
  std::vector<StorageType> sqrt_information;
};

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

std::vector<StorageType> NormalizeSqrtInformation(
    const float* prior_sqrt_information,
    const size_t num_priors,
    const double scale) {
  std::vector<StorageType> normalized(9 * num_priors);
  for (size_t prior = 0; prior < num_priors; ++prior) {
    Eigen::Map<PackedSqrtInformation>(normalized.data() + 9 * prior) =
        Eigen::Map<const InputSqrtInformation>(prior_sqrt_information +
                                               9 * prior)
            .cast<StorageType>() /
        static_cast<StorageType>(scale);
  }
  return normalized;
}

NormalizedPriors NormalizePriors(const float* prior_positions,
                                 const float* prior_sqrt_information,
                                 const size_t num_priors,
                                 const Sim3d& normalized_from_metric) {
  NormalizedPriors priors{
      std::vector<StorageType>(prior_positions,
                               prior_positions + 3 * num_priors),
      {},
  };
  bundle_adjustment_arrays::TransformPoints(priors.positions,
                                            normalized_from_metric);
  priors.sqrt_information = NormalizeSqrtInformation(
      prior_sqrt_information, num_priors, normalized_from_metric.scale());
  return priors;
}

ObservationFactorData BuildObservationFactors(
    const std::vector<CasparRowTrackSource>& sources,
    const std::vector<uint32_t>& source_track_offsets,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const std::vector<uint32_t>& selected_row_point_indices,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const size_t num_observations) {
  ObservationFactorData factors;
  factors.pose_indices.reserve(num_observations);
  factors.point_indices.reserve(num_observations);
  factors.sensor_indices.reserve(num_observations);
  factors.pixels.reserve(2 * num_observations);
  ForEachRowObservation(
      sources,
      source_track_offsets,
      point_track_offsets,
      point_track_indices,
      selected_row_point_indices,
      [&](const uint32_t point, const uint32_t image, const float* xy) {
        factors.pose_indices.push_back(image_frame_indices[image]);
        factors.point_indices.push_back(point);
        factors.sensor_indices.push_back(image_sensor_indices[image]);
        factors.pixels.push_back(xy[0]);
        factors.pixels.push_back(xy[1]);
      });
  return factors;
}

PriorFactorData BuildPriorFactors(const uint32_t* prior_frame_indices,
                                  NormalizedPriors priors) {
  return {
      std::vector<unsigned int>(
          prior_frame_indices,
          prior_frame_indices + priors.positions.size() / 3),
      std::move(priors.positions),
      std::move(priors.sqrt_information),
  };
}

}  // namespace

CasparRowBundleResult OptimizeRowCaspar(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const uint32_t* point_observation_counts,
    const uint32_t* selected_row_point_indices,
    const size_t num_selected_points,
    const std::vector<Rigid3d>& initial_rigs_from_world,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const size_t num_images,
    const std::vector<Rigid3d>& sensors_from_rig,
    const float* sensor_calibrations,
    const uint32_t* prior_frame_indices,
    const float* prior_positions,
    const float* prior_sqrt_information,
    const size_t num_priors,
    const CasparBundleAdjustmentOptions& options,
    bool refine_scale) {
#ifdef CASPAR_USE_DOUBLE
  LOG(FATAL_THROW) << "Caspar row bundle adjustment requires float precision";
  return {};
#else
  const Clock::time_point preparation_start = Clock::now();
  const std::vector<uint32_t> source_track_offsets =
      RowSourceTrackOffsets(sources);
  CasparRowPointSelection selected =
      InitializeRowPoints(sources,
                          source_track_offsets,
                          point_track_offsets,
                          point_track_indices,
                          selected_row_point_indices,
                          num_selected_points);
  size_t num_observations = 0;
  for (const uint32_t row_point : selected.row_point_indices) {
    num_observations += point_observation_counts[row_point];
  }
  ObservationFactorData observations =
      BuildObservationFactors(sources,
                              source_track_offsets,
                              point_track_offsets,
                              point_track_indices,
                              selected.row_point_indices,
                              image_frame_indices,
                              image_sensor_indices,
                              num_observations);
  const Sim3d normalized_from_metric =
      bundle_adjustment_arrays::ComputeNormalization(initial_rigs_from_world,
                                                     image_frame_indices,
                                                     image_sensor_indices,
                                                     num_images,
                                                     sensors_from_rig);
  bundle_adjustment_arrays::NormalizedProblem problem =
      bundle_adjustment_arrays::NormalizeProblem(
          normalized_from_metric,
          initial_rigs_from_world,
          selected.points.data(),
          selected.row_point_indices.size(),
          sensors_from_rig);
  NormalizedPriors normalized_priors = NormalizePriors(prior_positions,
                                                       prior_sqrt_information,
                                                       num_priors,
                                                       normalized_from_metric);
  PriorFactorData priors =
      BuildPriorFactors(prior_frame_indices, std::move(normalized_priors));
  std::vector<StorageType> rig_data =
      bundle_adjustment_arrays::PackPoses(problem.rigs_from_world);
  std::vector<StorageType> point_data = std::move(problem.points);
  const std::vector<StorageType> packed_sensor_calibrations =
      PackSensorCalibrations(problem.sensors_from_rig, sensor_calibrations);
  const double preparation_seconds = ElapsedSeconds(preparation_start);

  const Clock::time_point optimization_start = Clock::now();
  CasparSolverSizing sizing;
  sizing.num_pinhole_poses = initial_rigs_from_world.size();
  sizing.num_points = selected.row_point_indices.size();
  sizing.num_sensor_from_rig_log_scales = 1;
  sizing.num_row_fixed_rig_pinhole = observations.point_indices.size();
  sizing.num_row_fixed_rig_pinhole_sensor_calibrations =
      sensors_from_rig.size();
  sizing.num_fixed_rig_position_prior = num_priors;

  const std::vector<int> gpu_indices = CSVToVector<int>(options.gpu_index);
  const int gpu_index = gpu_indices.front();
  const size_t device_id =
      static_cast<size_t>(gpu_index >= 0 ? gpu_index : FindBestCudaDevice());
  auto solver =
      CreateSolver(CreateCasparSolverParameters(options), sizing, device_id);
  const StorageType log_scale = 0;
  solver.SetPinholePoseNodesFromStackedHost(
      rig_data.data(), 0, initial_rigs_from_world.size());
  solver.SetPointNodesFromStackedHost(
      point_data.data(), 0, selected.row_point_indices.size());
  solver.SetSensorFromRigLogScaleNodesFromStackedHost(&log_scale, 0, 1);
  solver.SetRowFixedRigPinholePoseIndicesFromHost(
      observations.pose_indices.data(), observations.pose_indices.size());
  solver.SetRowFixedRigPinholePointIndicesFromHost(
      observations.point_indices.data(), observations.point_indices.size());
  solver.SetRowFixedRigPinholeSensorCalibrationDataFromStackedHost(
      packed_sensor_calibrations.data(), 0, problem.sensors_from_rig.size());
  solver.SetRowFixedRigPinholeSensorCalibrationIndicesFromHost(
      observations.sensor_indices.data(), observations.sensor_indices.size());
  solver.SetRowFixedRigPinholePixelDataFromStackedHost(
      observations.pixels.data(), 0, observations.point_indices.size());
  const StorageType reprojection_loss_scale =
      options.fixed_rig_reprojection_loss_scale;
  solver.SetRowFixedRigPinholeReprojectionLossScaleDataFromStackedHost(
      &reprojection_loss_scale);
  solver.SetFixedRigPositionPriorPoseIndicesFromHost(priors.pose_indices.data(),
                                                     num_priors);
  solver.SetFixedRigPositionPriorPositionDataFromStackedHost(
      priors.positions.data(), 0, num_priors);
  solver.SetFixedRigPositionPriorSqrtInformationDataFromStackedHost(
      priors.sqrt_information.data(), 0, num_priors);
  const StorageType prior_position_loss_scale =
      options.prior_position_loss_scale;
  solver.SetFixedRigPositionPriorPositionLossScaleDataFromStackedHost(
      &prior_position_loss_scale);
  const unsigned int rotation_anchor_pose_index = *std::min_element(
      observations.pose_indices.begin(), observations.pose_indices.end());
  solver.SetRowFixedRigSchurTopology(observations.pose_indices,
                                     observations.point_indices,
                                     rotation_anchor_pose_index,
                                     refine_scale);
  solver.finish_indices();

  const caspar::SolveResult solve_result = solver.solve_rig_schur(
      /*print_progress=*/false,
      /*verbose_logging=*/options.collect_iteration_data);
  solver.GetPinholePoseNodesToStackedHost(
      rig_data.data(), 0, initial_rigs_from_world.size());
  solver.GetPointNodesToStackedHost(
      point_data.data(), 0, selected.row_point_indices.size());
  StorageType solved_log_scale;
  solver.GetSensorFromRigLogScaleNodesToStackedHost(&solved_log_scale, 0, 1);

  const Sim3d metric_from_normalized = Inverse(normalized_from_metric);
  std::vector<Rigid3d> solved_rigs_from_world =
      bundle_adjustment_arrays::UnpackPoses(rig_data);
  bundle_adjustment_arrays::TransformPoses(solved_rigs_from_world,
                                           metric_from_normalized);
  bundle_adjustment_arrays::TransformPoints(point_data, metric_from_normalized);

  CasparRowBundleResult result;
  result.rigs_from_world = std::move(solved_rigs_from_world);
  result.row_point_indices = std::move(selected.row_point_indices);
  result.points = std::move(point_data);
  result.sensor_from_rig_scale = std::exp(solved_log_scale);
  result.observation_count = observations.point_indices.size();
  result.preparation_seconds = preparation_seconds;
  result.optimization_seconds = ElapsedSeconds(optimization_start);
  result.summary = CasparBundleAdjustmentSummary::Create(solve_result);
  result.summary->num_residuals =
      static_cast<int>(2 * result.observation_count + 3 * num_priors);
  result.summary->allocation_size = solver.get_allocation_size();
  return result;
#endif
}

}  // namespace colmap
