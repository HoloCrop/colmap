#include "pycolmap/estimators/rig_calibration_initialization.h"

#include "colmap/estimators/rig_calibration_initialization.h"
#include "colmap/estimators/two_view_geometry.h"
#include "colmap/scene/database.h"
#include "colmap/util/hash_containers.h"

#include <pybind11/numpy.h>
#include <pybind11/stl/filesystem.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style>;
using Uint32Array = py::array_t<uint32_t, py::array::c_style>;

struct RigCalibrationInitializationResult {
  RigCalibrationInitializationSummary summary;
  DoubleArray rigs_from_group;
  size_t num_pairs = 0;
};

struct CalibrationImageLocation {
  size_t group_index;
  size_t frame_index;
  camera_t camera_id;
};

RigCalibrationInitializationResult InitializeRigCalibrationPacked(
    const RigCalibrationOptions& options,
    const rig_t rig_id,
    Reconstruction& reconstruction,
    const std::filesystem::path& database_path,
    const Uint32Array& image_ids,
    const Uint32Array& camera_ids,
    const DoubleArray& rigs_from_group,
    const DoubleArray& first_to_last_distances,
    const double distance_stddev) {
  THROW_CHECK_EQ(rigs_from_group.ndim(), 4);
  THROW_CHECK_EQ(rigs_from_group.shape(2), 3);
  THROW_CHECK_EQ(rigs_from_group.shape(3), 4);
  const ssize_t num_groups = rigs_from_group.shape(0);
  const ssize_t group_size = rigs_from_group.shape(1);
  THROW_CHECK_EQ(camera_ids.ndim(), 1);
  const ssize_t num_cameras = camera_ids.shape(0);
  THROW_CHECK_EQ(image_ids.ndim(), 3);
  THROW_CHECK_EQ(image_ids.shape(0), num_groups);
  THROW_CHECK_EQ(image_ids.shape(1), group_size);
  THROW_CHECK_EQ(image_ids.shape(2), num_cameras);
  THROW_CHECK_EQ(first_to_last_distances.ndim(), 1);
  THROW_CHECK_EQ(first_to_last_distances.shape(0), num_groups);

  const auto input_poses = rigs_from_group.unchecked<4>();
  const auto input_images = image_ids.unchecked<3>();
  const auto input_cameras = camera_ids.unchecked<1>();
  const auto input_distances = first_to_last_distances.unchecked<1>();
  RigCalibrationInitializationResult result;
  result.rigs_from_group = DoubleArray({num_groups, group_size, 3L, 4L});
  auto output_poses = result.rigs_from_group.mutable_unchecked<4>();

  py::gil_scoped_release release;
  FlatHashMap<image_t, CalibrationImageLocation> locations;
  std::vector<RigCalibrationInitializationGroup> groups(num_groups);
  for (ssize_t group_index = 0; group_index < num_groups; ++group_index) {
    auto& group = groups[group_index];
    group.first_to_last_distance = {input_distances(group_index),
                                    distance_stddev};
    group.rigs_from_group.reserve(group_size);
    for (ssize_t frame_index = 0; frame_index < group_size; ++frame_index) {
      Eigen::Matrix3x4d matrix;
      for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 4; ++column) {
          matrix(row, column) =
              input_poses(group_index, frame_index, row, column);
        }
      }
      group.rigs_from_group.push_back(Rigid3d::FromMatrix(matrix));
      for (ssize_t camera_index = 0; camera_index < num_cameras;
           ++camera_index) {
        THROW_CHECK(
            locations
                .emplace(
                    input_images(group_index, frame_index, camera_index),
                    CalibrationImageLocation{static_cast<size_t>(group_index),
                                             static_cast<size_t>(frame_index),
                                             input_cameras(camera_index)})
                .second);
      }
    }
  }

  const auto database = Database::Open(database_path);
  for (const auto& [pair_id, geometry] : database->ReadTwoViewGeometries()) {
    if (geometry.config != TwoViewGeometry::CALIBRATED ||
        !geometry.cam2_from_cam1.has_value()) {
      continue;
    }
    const auto [image_id1, image_id2] = PairIdToImagePair(pair_id);
    const auto first = locations.find(image_id1);
    const auto second = locations.find(image_id2);
    if (first == locations.end() || second == locations.end()) {
      continue;
    }
    const auto& location1 = first->second;
    const auto& location2 = second->second;
    THROW_CHECK_EQ(location1.group_index, location2.group_index);
    groups[location1.group_index].pairs.push_back({location1.frame_index,
                                                   location1.camera_id,
                                                   location2.frame_index,
                                                   location2.camera_id,
                                                   *geometry.cam2_from_cam1});
    ++result.num_pairs;
  }

  result.summary =
      InitializeRigCalibration(options, rig_id, groups, reconstruction);
  for (ssize_t group_index = 0; group_index < num_groups; ++group_index) {
    for (ssize_t frame_index = 0; frame_index < group_size; ++frame_index) {
      const Eigen::Matrix3x4d matrix =
          groups[group_index].rigs_from_group[frame_index].ToMatrix();
      for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 4; ++column) {
          output_poses(group_index, frame_index, row, column) =
              matrix(row, column);
        }
      }
    }
  }
  return result;
}

}  // namespace

void BindRigCalibrationInitialization(py::module& m) {
  using Summary = RigCalibrationInitializationSummary;
  py::classh<Summary>(m, "RigCalibrationInitializationSummary")
      .def_readonly("rotation_summary", &Summary::rotation_summary)
      .def_readonly("position_summary", &Summary::position_summary)
      .def_readonly("rotation_rank", &Summary::rotation_rank)
      .def_readonly("num_rotation_parameters",
                    &Summary::num_rotation_parameters)
      .def_readonly("position_rank", &Summary::position_rank)
      .def_readonly("num_position_parameters",
                    &Summary::num_position_parameters);
  using Result = RigCalibrationInitializationResult;
  py::classh<Result>(m, "RigCalibrationInitializationResult")
      .def_readonly("summary", &Result::summary)
      .def_readonly("rigs_from_group", &Result::rigs_from_group)
      .def_readonly("num_pairs", &Result::num_pairs);
  m.def(
      "initialize_rig_calibration",
      &InitializeRigCalibrationPacked,
      "options"_a,
      "rig_id"_a,
      "reconstruction"_a,
      "database_path"_a,
      "image_ids"_a.noconvert(),
      "camera_ids"_a.noconvert(),
      "rigs_from_group"_a.noconvert(),
      "first_to_last_distances"_a.noconvert(),
      "distance_stddev"_a,
      "Initialize camera extrinsics and group frame poses from independent "
      "two-view geometry, using measured endpoint distances for metric scale.");
}
