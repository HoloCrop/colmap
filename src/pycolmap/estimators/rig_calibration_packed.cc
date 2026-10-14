#include "pycolmap/estimators/rig_calibration_packed.h"

#include "colmap/estimators/rig_calibration.h"

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

using namespace colmap;
using namespace pybind11::literals;
namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style>;
using Uint32Array = py::array_t<uint32_t, py::array::c_style>;
using Uint64Array = py::array_t<uint64_t, py::array::c_style>;
using CameraIdArray = py::array_t<camera_t, py::array::c_style>;

std::unique_ptr<CeresRigCalibrator> CreateCeresRigCalibratorPacked(
    const RigCalibrationOptions& options,
    const rig_t rig_id,
    Reconstruction& reconstruction,
    const DoubleArray& rigs_from_group,
    const DoubleArray& first_to_last_distances,
    const double distance_stddev,
    const DoubleArray& distance_point_in_rig,
    const Uint64Array& group_track_offsets,
    const Uint64Array& track_observation_offsets,
    const DoubleArray& xyz,
    const Uint32Array& frame_indices,
    const CameraIdArray& camera_ids,
    const DoubleArray& xy) {
  THROW_CHECK_EQ(rigs_from_group.ndim(), 4);
  THROW_CHECK_GE(rigs_from_group.shape(1), 2);
  THROW_CHECK_EQ(rigs_from_group.shape(2), 3);
  THROW_CHECK_EQ(rigs_from_group.shape(3), 4);
  const ssize_t num_groups = rigs_from_group.shape(0);
  const ssize_t group_size = rigs_from_group.shape(1);

  THROW_CHECK_EQ(first_to_last_distances.ndim(), 1);
  THROW_CHECK_EQ(first_to_last_distances.shape(0), num_groups);
  THROW_CHECK_EQ(distance_point_in_rig.ndim(), 1);
  THROW_CHECK_EQ(distance_point_in_rig.shape(0), 3);
  const Eigen::Vector3d point_in_rig(distance_point_in_rig.at(0),
                                     distance_point_in_rig.at(1),
                                     distance_point_in_rig.at(2));
  THROW_CHECK_EQ(group_track_offsets.ndim(), 1);
  THROW_CHECK_EQ(group_track_offsets.shape(0), num_groups + 1);

  THROW_CHECK_EQ(xyz.ndim(), 2);
  THROW_CHECK_EQ(xyz.shape(1), 3);
  const ssize_t num_tracks = xyz.shape(0);
  THROW_CHECK_EQ(track_observation_offsets.ndim(), 1);
  THROW_CHECK_EQ(track_observation_offsets.shape(0), num_tracks + 1);

  THROW_CHECK_EQ(frame_indices.ndim(), 1);
  const ssize_t num_observations = frame_indices.shape(0);
  THROW_CHECK_EQ(camera_ids.ndim(), 1);
  THROW_CHECK_EQ(camera_ids.shape(0), num_observations);
  THROW_CHECK_EQ(xy.ndim(), 2);
  THROW_CHECK_EQ(xy.shape(0), num_observations);
  THROW_CHECK_EQ(xy.shape(1), 2);

  const auto group_track_offsets_view = group_track_offsets.unchecked<1>();
  const auto track_observation_offsets_view =
      track_observation_offsets.unchecked<1>();
  THROW_CHECK_EQ(group_track_offsets_view(0), 0);
  THROW_CHECK_EQ(group_track_offsets_view(num_groups),
                 static_cast<uint64_t>(num_tracks));
  for (ssize_t group_idx = 0; group_idx < num_groups; ++group_idx) {
    THROW_CHECK_LE(group_track_offsets_view(group_idx),
                   group_track_offsets_view(group_idx + 1));
  }
  THROW_CHECK_EQ(track_observation_offsets_view(0), 0);
  THROW_CHECK_EQ(track_observation_offsets_view(num_tracks),
                 static_cast<uint64_t>(num_observations));
  for (ssize_t track_idx = 0; track_idx < num_tracks; ++track_idx) {
    THROW_CHECK_LE(track_observation_offsets_view(track_idx),
                   track_observation_offsets_view(track_idx + 1));
  }

  const auto rigs_from_group_view = rigs_from_group.unchecked<4>();
  const auto first_to_last_distances_view =
      first_to_last_distances.unchecked<1>();
  const auto xyz_view = xyz.unchecked<2>();
  const auto frame_indices_view = frame_indices.unchecked<1>();
  const auto camera_ids_view = camera_ids.unchecked<1>();
  const auto xy_view = xy.unchecked<2>();

  py::gil_scoped_release release;
  std::vector<RigCalibrationGroup> groups;
  groups.reserve(num_groups);
  for (ssize_t group_idx = 0; group_idx < num_groups; ++group_idx) {
    RigCalibrationGroup group;
    group.rigs_from_group.reserve(group_size);
    for (ssize_t frame_idx = 0; frame_idx < group_size; ++frame_idx) {
      Eigen::Matrix3x4d matrix;
      for (size_t row = 0; row < 3; ++row) {
        for (size_t column = 0; column < 4; ++column) {
          matrix(row, column) =
              rigs_from_group_view(group_idx, frame_idx, row, column);
        }
      }
      group.rigs_from_group.push_back(Rigid3d::FromMatrix(matrix));
    }
    group.first_to_last_distance.distance =
        first_to_last_distances_view(group_idx);
    group.first_to_last_distance.stddev = distance_stddev;
    group.first_to_last_distance.point_in_rig = point_in_rig;

    const uint64_t track_begin = group_track_offsets_view(group_idx);
    const uint64_t track_end = group_track_offsets_view(group_idx + 1);
    group.tracks.reserve(track_end - track_begin);
    for (uint64_t track_idx = track_begin; track_idx < track_end; ++track_idx) {
      RigCalibrationTrack track;
      track.xyz = Eigen::Vector3d(xyz_view(track_idx, 0),
                                  xyz_view(track_idx, 1),
                                  xyz_view(track_idx, 2));
      const uint64_t observation_begin =
          track_observation_offsets_view(track_idx);
      const uint64_t observation_end =
          track_observation_offsets_view(track_idx + 1);
      track.observations.reserve(observation_end - observation_begin);
      for (uint64_t observation_idx = observation_begin;
           observation_idx < observation_end;
           ++observation_idx) {
        track.observations.push_back(
            {.frame_idx = frame_indices_view(observation_idx),
             .camera_id = camera_ids_view(observation_idx),
             .xy = Eigen::Vector2d(xy_view(observation_idx, 0),
                                   xy_view(observation_idx, 1))});
      }
      group.tracks.push_back(std::move(track));
    }
    groups.push_back(std::move(group));
  }
  return CreateCeresRigCalibrator(
      options, rig_id, std::move(groups), reconstruction);
}

}  // namespace

void BindRigCalibrationPacked(py::module& m) {
  m.def("create_ceres_rig_calibrator",
        &CreateCeresRigCalibratorPacked,
        "options"_a,
        "rig_id"_a,
        "reconstruction"_a,
        "rigs_from_group"_a,
        "first_to_last_distances"_a,
        "distance_stddev"_a,
        "distance_point_in_rig"_a,
        "group_track_offsets"_a,
        "track_observation_offsets"_a,
        "xyz"_a,
        "frame_indices"_a,
        "camera_ids"_a,
        "xy"_a,
        py::keep_alive<0, 3>());
}
