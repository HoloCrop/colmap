#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <Eigen/Core>

namespace colmap {

struct CasparRowTrackSource {
  const float* points;
  std::ptrdiff_t point_row_stride;
  std::ptrdiff_t point_column_stride;
  const int64_t* source_point_indices;
  const uint32_t* row_point_indices;
  const int64_t* observation_offsets;
  const int32_t* observation_image_indices;
  const float* observation_xy;
  const uint32_t* duplicate_observation_indices;
  const uint32_t* image_rows;
  size_t num_tracks;
  size_t num_images;
  size_t num_duplicate_observations;
};

inline Eigen::Vector3f ReadStridedVector3(const float* values,
                                          const std::ptrdiff_t row_stride,
                                          const std::ptrdiff_t column_stride,
                                          const int64_t row) {
  const float* row_values = values + row * row_stride;
  return {
      row_values[0], row_values[column_stride], row_values[2 * column_stride]};
}

inline Eigen::Vector3f RowSourcePoint(const CasparRowTrackSource& source,
                                      const int64_t point) {
  return ReadStridedVector3(source.points,
                            source.point_row_stride,
                            source.point_column_stride,
                            point);
}

struct CasparRowPointSelection {
  std::vector<uint32_t> row_point_indices;
  std::vector<float> points;
};

struct CasparRowTrackSelection {
  std::vector<uint32_t> point_indices;
  bool quota_truncated = false;
};

struct CasparRowTiers {
  std::vector<uint8_t> first_tier;
  bool quota_truncated = false;
};

struct CasparRowSectionDensities {
  std::vector<uint32_t> first_density;
  std::vector<size_t> observation_counts;
};

// Every whole-row selection at once: SelectCasparRowPoints at density d keeps
// exactly the points whose first_density <= d. point_counts[d] and
// observation_counts[d] (summing the given per-point observation counts) are
// that selection's sizes for d = 0..max density.
struct CasparRowPointDensities {
  std::vector<uint32_t> first_density;
  std::vector<uint64_t> point_counts;
  std::vector<uint64_t> observation_counts;
};

std::vector<uint32_t> RowSourceTrackOffsets(
    const std::vector<CasparRowTrackSource>& sources);

uint32_t CasparRowTrackQuotaPerImage(uint32_t tracks_per_spatial_cell);

CasparRowTiers AssignCasparRowTiers(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const float* sensor_dimensions,
    const bool* active_frame_mask,
    size_t num_row_points,
    size_t minimum_track_length,
    const uint32_t* density_tiers,
    size_t num_density_tiers);

CasparRowSectionDensities AssignCasparRowSectionDensities(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const float* sensor_dimensions,
    const bool* active_frame_masks,
    size_t num_sections,
    size_t num_frames,
    size_t num_row_points,
    size_t minimum_track_length);

CasparRowPointDensities AssignCasparRowPointDensities(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const float* sensor_dimensions,
    const bool* active_frame_mask,
    size_t num_frames,
    const uint32_t* point_observation_counts,
    size_t num_row_points,
    size_t minimum_track_length);

CasparRowTrackSelection SelectCasparRowPoints(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const float* sensor_dimensions,
    const bool* active_frame_mask,
    size_t num_row_points,
    size_t minimum_track_length,
    uint32_t tracks_per_spatial_cell);

CasparRowPointSelection InitializeRowPoints(
    const std::vector<CasparRowTrackSource>& sources,
    const std::vector<uint32_t>& source_track_offsets,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const uint32_t* row_point_indices,
    size_t num_points);

template <typename Callback>
void ForEachRowPointTrack(const std::vector<CasparRowTrackSource>& sources,
                          const std::vector<uint32_t>& source_track_offsets,
                          const uint32_t* point_track_offsets,
                          const uint32_t* point_track_indices,
                          const uint32_t row_point,
                          Callback callback) {
  const uint32_t point_track_start = point_track_offsets[row_point];
  const uint32_t point_track_end = point_track_offsets[row_point + 1];
  size_t source_index = 0;
  if (point_track_start < point_track_end) {
    source_index = static_cast<size_t>(
        std::upper_bound(source_track_offsets.begin() + 1,
                         source_track_offsets.end(),
                         point_track_indices[point_track_start]) -
        source_track_offsets.begin() - 1);
  }
  for (uint32_t point_track = point_track_start; point_track < point_track_end;
       ++point_track) {
    const uint32_t global_track = point_track_indices[point_track];
    while (global_track >= source_track_offsets[source_index + 1]) {
      ++source_index;
    }
    const uint32_t track = global_track - source_track_offsets[source_index];
    callback(source_index, sources[source_index], track);
  }
}

template <typename Callback>
void ForEachRowTrackObservation(const CasparRowTrackSource& source,
                                const uint32_t track,
                                Callback callback) {
  const int64_t observation_start = source.observation_offsets[track];
  const int64_t observation_end = source.observation_offsets[track + 1];
  const uint32_t* duplicate = std::lower_bound(
      source.duplicate_observation_indices,
      source.duplicate_observation_indices + source.num_duplicate_observations,
      static_cast<uint32_t>(observation_start));
  const uint32_t* duplicate_end = std::lower_bound(
      duplicate,
      source.duplicate_observation_indices + source.num_duplicate_observations,
      static_cast<uint32_t>(observation_end));
  for (int64_t observation = observation_start; observation < observation_end;
       ++observation) {
    if (duplicate != duplicate_end && *duplicate == observation) {
      ++duplicate;
      continue;
    }
    const uint32_t image =
        source.image_rows[source.observation_image_indices[observation]];
    callback(
        static_cast<uint32_t>(source.observation_image_indices[observation]),
        image,
        source.observation_xy + 2 * observation);
  }
}

template <typename Callback>
void ForEachRowObservation(
    const std::vector<CasparRowTrackSource>& sources,
    const std::vector<uint32_t>& source_track_offsets,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const std::vector<uint32_t>& selected_row_point_indices,
    Callback callback) {
  for (size_t point = 0; point < selected_row_point_indices.size(); ++point) {
    ForEachRowPointTrack(
        sources,
        source_track_offsets,
        point_track_offsets,
        point_track_indices,
        selected_row_point_indices[point],
        [&](const size_t,
            const CasparRowTrackSource& source,
            const uint32_t track) {
          ForEachRowTrackObservation(
              source,
              track,
              [&](const uint32_t, const uint32_t image, const float* xy) {
                callback(static_cast<uint32_t>(point), image, xy);
              });
        });
  }
}

}  // namespace colmap
