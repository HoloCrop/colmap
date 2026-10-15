#include "colmap/estimators/surface_bundle_adjustment_caspar.h"

#include "colmap/estimators/bundle_adjustment_arrays_caspar.h"
#include "colmap/estimators/caspar/caspar_model_adapter.h"
#include "colmap/estimators/caspar/surface_normalization.h"
#include "colmap/geometry/sim3.h"
#include "colmap/util/cuda.h"
#include "colmap/util/logging.h"
#include "colmap/util/misc.h"

#include <algorithm>
#include <memory>
#include <utility>

#include <cuda_runtime.h>

namespace colmap {
namespace {

void CheckCuda(const cudaError_t status) {
  THROW_CHECK_EQ(status, cudaSuccess) << cudaGetErrorString(status);
}

struct CudaFree {
  void operator()(float* data) const { cudaFree(data); }
};

using DeviceFloats = std::unique_ptr<float, CudaFree>;

DeviceFloats AllocateFloats(const size_t num) {
  float* data = nullptr;
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&data),
                       std::max<size_t>(num, 1) * sizeof(float)));
  return DeviceFloats(data);
}

std::vector<unsigned int> CopyIndicesToHost(const uint32_t* device_indices,
                                            const size_t num) {
  std::vector<unsigned int> host(num);
  CheckCuda(cudaMemcpy(host.data(),
                       device_indices,
                       num * sizeof(unsigned int),
                       cudaMemcpyDeviceToHost));
  return host;
}

// Every (frame, sensor) image, for the camera-centre normalization.
Sim3d ComputeRigNormalization(const std::vector<Rigid3d>& rigs_from_world,
                              const std::vector<Rigid3d>& sensors_from_rig) {
  const size_t num_images = rigs_from_world.size() * sensors_from_rig.size();
  std::vector<uint32_t> image_frame_indices(num_images);
  std::vector<uint32_t> image_sensor_indices(num_images);
  for (size_t image = 0; image < num_images; ++image) {
    image_frame_indices[image] = image / sensors_from_rig.size();
    image_sensor_indices[image] = image % sensors_from_rig.size();
  }
  return bundle_adjustment_arrays::ComputeNormalization(
      rigs_from_world,
      image_frame_indices.data(),
      image_sensor_indices.data(),
      num_images,
      sensors_from_rig);
}

int DeviceIndex(const CasparBundleAdjustmentOptions& options) {
  const int gpu_index = CSVToVector<int>(options.gpu_index).front();
  return gpu_index >= 0 ? gpu_index : FindBestCudaDevice();
}

CasparSolverSizing Sizing(const size_t num_frames,
                          const size_t num_points,
                          const size_t num_sensors,
                          const size_t num_observations,
                          const size_t max_surface_factors) {
  CasparSolverSizing sizing;
  sizing.num_pinhole_poses = num_frames;
  sizing.num_points = num_points;
  sizing.num_sensor_from_rig_log_scales = 1;
  sizing.num_row_fixed_rig_pinhole = num_observations;
  sizing.num_row_fixed_rig_pinhole_sensor_calibrations = num_sensors;
  sizing.num_rig_surface = max_surface_factors;
  return sizing;
}

}  // namespace

#ifdef CASPAR_USE_DOUBLE
class CasparSurfaceBundleAdjuster::Impl {
 public:
  int device_id = -1;

  CasparSurfaceBundleResult Solve(const std::vector<Rigid3d>&,
                                  float*,
                                  const CasparSurfaceFactors&) {
    LOG(FATAL_THROW) << "Caspar surface bundle adjustment requires float "
                        "precision";
    return {};
  }
};
#else
class CasparSurfaceBundleAdjuster::Impl {
 public:
  Impl(const std::vector<Rigid3d>& initial_rigs_from_world,
       const std::vector<Rigid3d>& sensors_from_rig,
       const float* sensor_calibrations,
       const CasparSurfaceObservations& observations,
       const size_t num_points,
       const size_t max_surface_factors,
       const CasparBundleAdjustmentOptions& options)
      : device_id(DeviceIndex(options)),
        normalized_from_metric(
            ComputeRigNormalization(initial_rigs_from_world, sensors_from_rig)),
        num_frames(initial_rigs_from_world.size()),
        num_points(num_points),
        num_observations(observations.count),
        solver(CreateSolver(CreateCasparSolverParameters(options),
                            Sizing(initial_rigs_from_world.size(),
                                   num_points,
                                   sensors_from_rig.size(),
                                   observations.count,
                                   max_surface_factors),
                            device_id)),
        points(AllocateFloats(3 * num_points)),
        surface_rig_points(AllocateFloats(3 * max_surface_factors)),
        surface_positions(AllocateFloats(3 * max_surface_factors)),
        surface_sqrt_weights(AllocateFloats(max_surface_factors)) {
    std::vector<Rigid3d> normalized_sensors = sensors_from_rig;
    for (Rigid3d& sensor_from_rig : normalized_sensors) {
      sensor_from_rig.translation() *= normalized_from_metric.scale();
    }
    const std::vector<StorageType> calibrations =
        bundle_adjustment_arrays::PackSensorCalibrations(normalized_sensors,
                                                         sensor_calibrations);
    const StorageType log_scale = 0;
    solver.SetSensorFromRigLogScaleNodesFromStackedHost(&log_scale, 0, 1);
    solver.SetRowFixedRigPinholePoseIndicesFromDevice(observations.frame_indices,
                                                      observations.count);
    solver.SetRowFixedRigPinholePointIndicesFromDevice(
        observations.point_indices, observations.count);
    solver.SetRowFixedRigPinholeSensorCalibrationDataFromStackedHost(
        calibrations.data(), 0, normalized_sensors.size());
    solver.SetRowFixedRigPinholeSensorCalibrationIndicesFromDevice(
        observations.sensor_indices, observations.count);
    solver.SetRowFixedRigPinholePixelDataFromStackedDevice(
        observations.pixels, 0, observations.count);
    const StorageType reprojection_loss_scale =
        options.fixed_rig_reprojection_loss_scale;
    solver.SetRowFixedRigPinholeReprojectionLossScaleDataFromStackedHost(
        &reprojection_loss_scale);
    solver.SetRowFixedRigSchurTopology(
        CopyIndicesToHost(observations.frame_indices, observations.count),
        CopyIndicesToHost(observations.point_indices, observations.count),
        /*rotation_anchor_pose_index=*/-1,
        /*refine_scale=*/false);
  }

  CasparSurfaceBundleResult Solve(const std::vector<Rigid3d>& rigs_from_world,
                                  float* metric_points,
                                  const CasparSurfaceFactors& factors) {
    THROW_CHECK_EQ(rigs_from_world.size(), num_frames);
    CheckCuda(cudaSetDevice(device_id));
    const float scale = static_cast<float>(normalized_from_metric.scale());
    const Eigen::Vector3f translation =
        normalized_from_metric.translation().cast<float>();

    std::vector<Rigid3d> normalized_rigs = rigs_from_world;
    bundle_adjustment_arrays::TransformPoses(normalized_rigs,
                                             normalized_from_metric);
    std::vector<StorageType> rig_data =
        bundle_adjustment_arrays::PackPoses(normalized_rigs);
    solver.SetPinholePoseNodesFromStackedHost(rig_data.data(), 0, num_frames);
    AffineTransformPointsDevice(metric_points,
                                points.get(),
                                num_points,
                                scale,
                                translation.x(),
                                translation.y(),
                                translation.z());
    solver.SetPointNodesFromStackedDevice(points.get(), 0, num_points);

    solver.SetRigSurfaceNum(factors.count);
    solver.SetRigSurfacePoseIndicesFromDevice(factors.frame_indices,
                                              factors.count);
    // Rig coordinates scale with the scene.
    ScaleDevice(factors.rig_points,
                surface_rig_points.get(),
                3 * factors.count,
                scale);
    solver.SetRigSurfaceRigPointDataFromStackedDevice(
        surface_rig_points.get(), 0, factors.count);
    AffineTransformPointsDevice(factors.positions,
                                surface_positions.get(),
                                factors.count,
                                scale,
                                translation.x(),
                                translation.y(),
                                translation.z());
    solver.SetRigSurfacePositionDataFromStackedDevice(
        surface_positions.get(), 0, factors.count);
    // Distances grow by the scale, so sqrt weights shrink by it to keep the
    // residuals in metric units.
    ScaleDevice(factors.sqrt_weights,
                surface_sqrt_weights.get(),
                factors.count,
                1.0f / scale);
    solver.SetRigSurfaceSqrtWeightDataFromStackedDevice(
        surface_sqrt_weights.get(), 0, factors.count);
    const StorageType robust[2] = {
        static_cast<StorageType>(factors.robust_scale * scale),
        static_cast<StorageType>(factors.margin * scale),
    };
    solver.SetRigSurfaceRobustDataFromStackedHost(robust);
    solver.finish_indices();

    const caspar::SolveResult solve_result = solver.solve_rig_schur();

    solver.GetPinholePoseNodesToStackedHost(rig_data.data(), 0, num_frames);
    std::vector<Rigid3d> solved =
        bundle_adjustment_arrays::UnpackPoses(rig_data);
    bundle_adjustment_arrays::TransformPoses(solved,
                                             Inverse(normalized_from_metric));
    solver.GetPointNodesToStackedDevice(points.get(), 0, num_points);
    AffineTransformPointsDevice(points.get(),
                                metric_points,
                                num_points,
                                1.0f / scale,
                                -translation.x() / scale,
                                -translation.y() / scale,
                                -translation.z() / scale);

    CasparSurfaceBundleResult result;
    result.rigs_from_world = std::move(solved);
    result.summary = CasparBundleAdjustmentSummary::Create(solve_result);
    result.summary->num_residuals =
        static_cast<int>(2 * num_observations + 3 * factors.count);
    result.summary->allocation_size = solver.get_allocation_size();
    return result;
  }

  const int device_id;
  const Sim3d normalized_from_metric;
  const size_t num_frames;
  const size_t num_points;
  const size_t num_observations;
  caspar::GraphSolver solver;
  DeviceFloats points;
  DeviceFloats surface_rig_points;
  DeviceFloats surface_positions;
  DeviceFloats surface_sqrt_weights;
};
#endif

#ifdef CASPAR_USE_DOUBLE
CasparSurfaceBundleAdjuster::CasparSurfaceBundleAdjuster(
    const std::vector<Rigid3d>&,
    const std::vector<Rigid3d>&,
    const float*,
    const CasparSurfaceObservations&,
    size_t,
    size_t,
    const CasparBundleAdjustmentOptions&) {
  LOG(FATAL_THROW) << "Caspar surface bundle adjustment requires float "
                      "precision";
}
#else
CasparSurfaceBundleAdjuster::CasparSurfaceBundleAdjuster(
    const std::vector<Rigid3d>& initial_rigs_from_world,
    const std::vector<Rigid3d>& sensors_from_rig,
    const float* sensor_calibrations,
    const CasparSurfaceObservations& observations,
    const size_t num_points,
    const size_t max_surface_factors,
    const CasparBundleAdjustmentOptions& options)
    : impl_(std::make_unique<Impl>(initial_rigs_from_world,
                                   sensors_from_rig,
                                   sensor_calibrations,
                                   observations,
                                   num_points,
                                   max_surface_factors,
                                   options)) {}
#endif

CasparSurfaceBundleAdjuster::~CasparSurfaceBundleAdjuster() = default;

int CasparSurfaceBundleAdjuster::DeviceId() const { return impl_->device_id; }

CasparSurfaceBundleResult CasparSurfaceBundleAdjuster::Solve(
    const std::vector<Rigid3d>& rigs_from_world,
    float* points,
    const CasparSurfaceFactors& factors) {
  return impl_->Solve(rigs_from_world, points, factors);
}

}  // namespace colmap
