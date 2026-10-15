#ifdef CASPAR_ENABLED
#include "colmap/estimators/surface_bundle_adjustment_caspar.h"
#endif

#include "colmap/util/logging.h"

#include "pycolmap/helpers.h"
#include "pycolmap/pybind11_extension.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#ifdef CASPAR_ENABLED
#include <cuda_runtime.h>
#endif

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

#ifdef CASPAR_ENABLED

int PointerDevice(const void* data) {
  cudaPointerAttributes attributes;
  THROW_CHECK_EQ(cudaPointerGetAttributes(&attributes, data), cudaSuccess);
  THROW_CHECK_EQ(attributes.type, cudaMemoryTypeDevice);
  return attributes.device;
}

// A C-contiguous CUDA array exposed through __cuda_array_interface__ (for
// example a torch CUDA tensor) on the solver's device. A leading dimension of
// -1 in `shape` accepts any length.
template <typename T>
struct CudaArray {
  T* data;
  size_t length;
};

template <typename T>
CudaArray<T> ReadCudaArray(const py::handle& array,
                           const std::string& name,
                           const std::string& typestr,
                           const std::vector<ssize_t>& shape,
                           const int device_id,
                           const bool writable) {
  THROW_CHECK(py::hasattr(array, "__cuda_array_interface__"))
      << name << " must be a CUDA array (e.g. a torch CUDA tensor)";
  const py::dict interface = array.attr("__cuda_array_interface__");
  THROW_CHECK_EQ(interface["typestr"].cast<std::string>(), typestr)
      << name << " dtype";
  const auto array_shape = interface["shape"].cast<std::vector<ssize_t>>();
  THROW_CHECK_EQ(array_shape.size(), shape.size()) << name << " dimensions";
  for (size_t dim = 0; dim < shape.size(); ++dim) {
    THROW_CHECK(shape[dim] < 0 || shape[dim] == array_shape[dim])
        << name << " shape";
  }
  THROW_CHECK(!interface.contains("strides") || interface["strides"].is_none())
      << name << " must be C-contiguous";
  const auto pointer = interface["data"].cast<std::pair<uintptr_t, bool>>();
  THROW_CHECK(!writable || !pointer.second) << name << " must be writable";
  T* data = reinterpret_cast<T*>(pointer.first);
  if (array_shape.front() > 0) {
    THROW_CHECK_EQ(PointerDevice(data), device_id)
        << name << " is not on the solver's CUDA device";
  }
  return {data, static_cast<size_t>(array_shape.front())};
}

class PySurfaceBundleAdjuster {
 public:
  PySurfaceBundleAdjuster(const std::vector<Rigid3d>& initial_rigs_from_world,
                          const std::vector<Rigid3d>& sensors_from_rig,
                          const py::array_t<float, py::array::c_style>&
                              sensor_calibrations,
                          const py::object& observation_points,
                          const py::object& observation_frames,
                          const py::object& observation_sensors,
                          const py::object& observation_pixels,
                          const size_t num_points,
                          const size_t max_surface_factors,
                          const CasparBundleAdjustmentOptions& options)
      : num_points_(num_points) {
    THROW_CHECK_EQ(sensor_calibrations.ndim(), 2);
    THROW_CHECK_EQ(sensor_calibrations.shape(0), sensors_from_rig.size());
    THROW_CHECK_EQ(sensor_calibrations.shape(1), 4);
    // The solver runs on the device holding the observations.
    const py::dict interface = observation_points.attr("__cuda_array_interface__");
    const int device_id = PointerDevice(reinterpret_cast<const void*>(
        interface["data"].cast<std::pair<uintptr_t, bool>>().first));
    CasparBundleAdjustmentOptions device_options = options;
    device_options.gpu_index = std::to_string(device_id);
    const auto points = ReadCudaArray<const uint32_t>(
        observation_points, "observation_points", "<u4", {-1}, device_id, false);
    const size_t count = points.length;
    CasparSurfaceObservations observations{
        points.data,
        ReadCudaArray<const uint32_t>(observation_frames,
                                      "observation_frames",
                                      "<u4",
                                      {static_cast<ssize_t>(count)},
                                      device_id,
                                      false)
            .data,
        ReadCudaArray<const uint32_t>(observation_sensors,
                                      "observation_sensors",
                                      "<u4",
                                      {static_cast<ssize_t>(count)},
                                      device_id,
                                      false)
            .data,
        ReadCudaArray<const float>(observation_pixels,
                                   "observation_pixels",
                                   "<f4",
                                   {static_cast<ssize_t>(count), 2},
                                   device_id,
                                   false)
            .data,
        count,
    };
    adjuster_ = std::make_unique<CasparSurfaceBundleAdjuster>(
        initial_rigs_from_world,
        sensors_from_rig,
        sensor_calibrations.data(),
        observations,
        num_points,
        max_surface_factors,
        device_options);
  }

  CasparSurfaceBundleResult Solve(const std::vector<Rigid3d>& rigs_from_world,
                                  const py::object& points,
                                  const py::object& surface_frames,
                                  const py::object& surface_rig_points,
                                  const py::object& surface_positions,
                                  const py::object& surface_sqrt_weights,
                                  const double surface_robust_scale,
                                  const double surface_margin) {
    THROW_CHECK_GT(surface_robust_scale, 0.0);
    THROW_CHECK_GE(surface_margin, 0.0);
    const int device_id = adjuster_->DeviceId();
    float* point_data =
        ReadCudaArray<float>(points,
                             "points",
                             "<f4",
                             {static_cast<ssize_t>(num_points_), 3},
                             device_id,
                             true)
            .data;
    const auto indices = ReadCudaArray<const uint32_t>(
        surface_frames, "surface_frames", "<u4", {-1}, device_id, false);
    const ssize_t count = static_cast<ssize_t>(indices.length);
    const CasparSurfaceFactors factors{
        indices.data,
        ReadCudaArray<const float>(surface_rig_points,
                                   "surface_rig_points",
                                   "<f4",
                                   {count, 3},
                                   device_id,
                                   false)
            .data,
        ReadCudaArray<const float>(surface_positions,
                                   "surface_positions",
                                   "<f4",
                                   {count, 3},
                                   device_id,
                                   false)
            .data,
        ReadCudaArray<const float>(surface_sqrt_weights,
                                   "surface_sqrt_weights",
                                   "<f4",
                                   {count},
                                   device_id,
                                   false)
            .data,
        indices.length,
        surface_robust_scale,
        surface_margin,
    };
    py::gil_scoped_release release;
    return adjuster_->Solve(rigs_from_world, point_data, factors);
  }

 private:
  size_t num_points_;
  std::unique_ptr<CasparSurfaceBundleAdjuster> adjuster_;
};

#endif

}  // namespace

void BindSurfaceBundleAdjustment(py::module& m) {
#ifdef CASPAR_ENABLED
  py::classh<CasparSurfaceBundleResult>(m, "CasparSurfaceBundleResult")
      .def_readonly("rigs_from_world",
                    &CasparSurfaceBundleResult::rigs_from_world)
      .def_readonly("summary", &CasparSurfaceBundleResult::summary);

  py::classh<PySurfaceBundleAdjuster>(m, "CasparSurfaceBundleAdjuster")
      .def(py::init<const std::vector<Rigid3d>&,
                    const std::vector<Rigid3d>&,
                    const py::array_t<float, py::array::c_style>&,
                    const py::object&,
                    const py::object&,
                    const py::object&,
                    const py::object&,
                    size_t,
                    size_t,
                    const CasparBundleAdjustmentOptions&>(),
           "initial_rigs_from_world"_a,
           "sensors_from_rig"_a,
           "sensor_calibrations"_a.noconvert(),
           "observation_points"_a,
           "observation_frames"_a,
           "observation_sensors"_a,
           "observation_pixels"_a,
           "num_points"_a,
           "max_surface_factors"_a,
           "options"_a = CasparBundleAdjustmentOptions(),
           "Fixed-rig bundle adjustment of one trajectory and its points "
           "against a fixed surface. Observations (uint32 point, frame and "
           "sensor indices, float32 pixels) are CUDA arrays uploaded once; "
           "sensor_calibrations is host float32 (sensors, 4) fx fy cx cy. "
           "Calibration and rig scale stay fixed; the surface fixes the "
           "gauge.")
      .def("solve",
           &PySurfaceBundleAdjuster::Solve,
           "rigs_from_world"_a,
           "points"_a,
           "surface_frames"_a,
           "surface_rig_points"_a,
           "surface_positions"_a,
           "surface_sqrt_weights"_a,
           "surface_robust_scale"_a,
           "surface_margin"_a,
           "Solve from the given rig poses and float32 CUDA points "
           "(updated in place). Each surface factor ties a point fixed in a "
           "frame's rig coordinates to a fixed world position: its offset "
           "beyond the margin, through Geman-McClure of the robust scale "
           "(metres), times its sqrt weight.");
#endif
}
