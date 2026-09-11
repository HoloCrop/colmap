#include "colmap/sensor/source_coordinate_restoration.h"

#include "colmap/util/logging.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;
using namespace colmap;

void BindSourceCoordinateRestoration(py::module& module) {
  py::class_<SourceCoordinateCamera>(module, "SourceCoordinateCamera")
      .def(py::init<const Eigen::Vector4d&,
                    const Eigen::Vector4d&,
                    const std::vector<double>&>(),
           "prepared_intrinsics"_a,
           "source_intrinsics"_a,
           "distortion"_a);
  module.def(
      "restore_source_coordinates",
      [](const std::vector<SourceCoordinateCamera>& cameras,
         py::array_t<int32_t, py::array::c_style> camera_indices,
         py::array_t<float, py::array::c_style> pixels) {
        THROW_CHECK_EQ(camera_indices.ndim(), 1);
        THROW_CHECK_EQ(pixels.ndim(), 2);
        THROW_CHECK_EQ(pixels.shape(1), 2);
        THROW_CHECK_EQ(pixels.shape(0), camera_indices.shape(0));
        const auto indices = camera_indices.unchecked<1>();
        const auto points = pixels.unchecked<2>();
        for (py::ssize_t index = 0; index < pixels.shape(0); ++index) {
          THROW_CHECK_GE(indices(index), 0);
          THROW_CHECK_LT(static_cast<size_t>(indices(index)), cameras.size());
        }
        py::array_t<float> restored(
            std::vector<py::ssize_t>{pixels.shape(0), 2});
        auto output = restored.mutable_unchecked<2>();
        {
          py::gil_scoped_release release;
          for (py::ssize_t index = 0; index < pixels.shape(0); ++index) {
            const auto point = cameras[indices(index)].Restore(
                Eigen::Vector2f(points(index, 0), points(index, 1)));
            output(index, 0) = point.x();
            output(index, 1) = point.y();
          }
        }
        return restored;
      },
      "cameras"_a,
      "camera_indices"_a,
      "pixels"_a,
      "Restore alignment track coordinates in one pass without Jacobians.");
}
