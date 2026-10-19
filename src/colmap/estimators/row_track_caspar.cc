#include "colmap/estimators/row_track_caspar.h"

#include <algorithm>
#include <limits>
#include <numeric>

#include <Eigen/Core>

namespace colmap {
namespace {

constexpr uint32_t kSpatialGridSide = 4;
constexpr uint32_t kSpatialCellCount = kSpatialGridSide * kSpatialGridSide;

size_t RowTrackLength(const CasparRowTrackSource& source,
                      const uint32_t track) {
  return static_cast<size_t>(source.observation_offsets[track + 1] -
                             source.observation_offsets[track]);
}

std::vector<size_t> RowPointTrackLengths(
    const std::vector<CasparRowTrackSource>& sources,
    const size_t num_row_points) {
  std::vector<size_t> lengths(num_row_points);
  for (const CasparRowTrackSource& source : sources) {
    for (uint32_t track = 0; track < source.num_tracks; ++track) {
      lengths[source.row_point_indices[track]] += RowTrackLength(source, track);
    }
  }
  return lengths;
}

void FillTrackOrder(const CasparRowTrackSource& source,
                    const std::vector<size_t>& row_point_track_lengths,
                    const size_t minimum_track_length,
                    std::vector<uint32_t>& bucket_end,
                    std::vector<uint32_t>& ordered_tracks) {
  uint32_t end = 0;
  for (auto bucket = bucket_end.rbegin(); bucket != bucket_end.rend();
       ++bucket) {
    end += *bucket;
    *bucket = end;
  }
  ordered_tracks.resize(end);
  for (size_t track = source.num_tracks; track > 0; --track) {
    const uint32_t track_index = static_cast<uint32_t>(track - 1);
    const size_t length =
        row_point_track_lengths[source.row_point_indices[track_index]];
    if (length >= minimum_track_length) {
      ordered_tracks[--bucket_end[length]] = track_index;
    }
  }
}

}  // namespace

uint32_t CasparRowTrackQuotaPerImage(const uint32_t tracks_per_spatial_cell) {
  return tracks_per_spatial_cell * kSpatialCellCount;
}

std::vector<uint32_t> RowSourceTrackOffsets(
    const std::vector<CasparRowTrackSource>& sources) {
  std::vector<uint32_t> offsets(sources.size() + 1);
  for (size_t source = 0; source < sources.size(); ++source) {
    offsets[source + 1] =
        offsets[source] + static_cast<uint32_t>(sources[source].num_tracks);
  }
  return offsets;
}

CasparRowTiers AssignCasparRowTiers(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const float* sensor_dimensions,
    const bool* active_frame_mask,
    const size_t num_row_points,
    const size_t minimum_track_length,
    const uint32_t* density_tiers,
    const size_t num_density_tiers) {
  const uint8_t no_tier = static_cast<uint8_t>(num_density_tiers);
  CasparRowTiers assignment{std::vector<uint8_t>(num_row_points, no_tier),
                            false};
  const uint32_t maximum_density = density_tiers[num_density_tiers - 1];
  const std::vector<size_t> row_point_track_lengths =
      RowPointTrackLengths(sources, num_row_points);
  std::vector<bool> denied_point(maximum_density > 0 ? num_row_points : 0,
                                 false);
  std::vector<uint32_t> stratum_count;
  std::vector<uint32_t> bucket_end;
  std::vector<uint32_t> ordered_tracks;
  for (const CasparRowTrackSource& source : sources) {
    bool source_is_active = false;
    for (size_t image = 0; image < source.num_images; ++image) {
      if (active_frame_mask[image_frame_indices[source.image_rows[image]]]) {
        source_is_active = true;
        break;
      }
    }
    if (!source_is_active) {
      continue;
    }
    bucket_end.clear();
    for (uint32_t track = 0; track < source.num_tracks; ++track) {
      const size_t length =
          row_point_track_lengths[source.row_point_indices[track]];
      if (length < minimum_track_length) {
        continue;
      }
      if (maximum_density == 0) {
        assignment.quota_truncated = true;
      } else {
        if (bucket_end.size() <= length) {
          bucket_end.resize(length + 1);
        }
        ++bucket_end[length];
      }
    }
    if (maximum_density == 0) {
      continue;
    }

    stratum_count.assign(source.num_images * kSpatialCellCount, uint32_t{0});
    FillTrackOrder(source,
                   row_point_track_lengths,
                   minimum_track_length,
                   bucket_end,
                   ordered_tracks);
    for (const uint32_t track : ordered_tracks) {
      const uint32_t row_point = source.row_point_indices[track];
      uint32_t first_density = 0;
      bool admitted = false;
      ForEachRowTrackObservation(
          source,
          track,
          [&](const uint32_t source_image,
              const uint32_t image,
              const float* xy) {
            if (!active_frame_mask[image_frame_indices[image]]) {
              return;
            }
            const float* dimensions =
                sensor_dimensions + 2 * image_sensor_indices[image];
            if (xy[0] < 0.0f || xy[0] >= dimensions[0] || xy[1] < 0.0f ||
                xy[1] >= dimensions[1]) {
              return;
            }
            const uint32_t cell_x =
                static_cast<uint32_t>(xy[0] * kSpatialGridSide / dimensions[0]);
            const uint32_t cell_y =
                static_cast<uint32_t>(xy[1] * kSpatialGridSide / dimensions[1]);
            const uint32_t stratum = source_image * kSpatialCellCount +
                                     cell_y * kSpatialGridSide + cell_x;
            uint32_t& count = stratum_count[stratum];
            if (count < maximum_density) {
              ++count;
              if (!admitted || count < first_density) {
                first_density = count;
                admitted = true;
              }
            } else {
              denied_point[row_point] = true;
            }
          });
      if (admitted) {
        const uint8_t first_tier = static_cast<uint8_t>(
            std::lower_bound(density_tiers,
                             density_tiers + num_density_tiers,
                             first_density) -
            density_tiers);
        assignment.first_tier[row_point] =
            std::min(assignment.first_tier[row_point], first_tier);
      }
    }
  }
  if (maximum_density > 0) {
    for (uint32_t row_point = 0; row_point < num_row_points; ++row_point) {
      if (denied_point[row_point] &&
          assignment.first_tier[row_point] == no_tier) {
        assignment.quota_truncated = true;
        break;
      }
    }
  }
  return assignment;
}

CasparRowSectionDensities AssignCasparRowSectionDensities(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const float* sensor_dimensions,
    const bool* active_frame_masks,
    const size_t num_sections,
    const size_t num_frames,
    const size_t num_row_points,
    const size_t minimum_track_length) {
  const uint32_t unselected = std::numeric_limits<uint32_t>::max();
  CasparRowSectionDensities assignment{
      std::vector<uint32_t>(num_sections * num_row_points, unselected),
      RowPointTrackLengths(sources, num_row_points)};

  std::vector<uint32_t> frame_section_offsets(num_frames + 1);
  for (size_t frame = 0; frame < num_frames; ++frame) {
    for (size_t section = 0; section < num_sections; ++section) {
      frame_section_offsets[frame + 1] +=
          active_frame_masks[section * num_frames + frame];
    }
    frame_section_offsets[frame + 1] += frame_section_offsets[frame];
  }
  std::vector<uint32_t> frame_sections(frame_section_offsets.back());
  std::vector<uint32_t> frame_section_write = frame_section_offsets;
  for (uint32_t frame = 0; frame < num_frames; ++frame) {
    for (uint32_t section = 0; section < num_sections; ++section) {
      if (active_frame_masks[section * num_frames + frame]) {
        frame_sections[frame_section_write[frame]++] = section;
      }
    }
  }

  std::vector<uint32_t> bucket_end;
  std::vector<uint32_t> ordered_tracks;
  std::vector<uint32_t> first_density(num_sections);
  for (const CasparRowTrackSource& source : sources) {
    bucket_end.clear();
    for (uint32_t track = 0; track < source.num_tracks; ++track) {
      const size_t length =
          assignment.observation_counts[source.row_point_indices[track]];
      if (length >= minimum_track_length) {
        if (bucket_end.size() <= length) {
          bucket_end.resize(length + 1);
        }
        ++bucket_end[length];
      }
    }
    FillTrackOrder(source,
                   assignment.observation_counts,
                   minimum_track_length,
                   bucket_end,
                   ordered_tracks);
    std::vector<uint32_t> stratum_counts(num_sections * source.num_images *
                                         kSpatialCellCount);
    for (const uint32_t track : ordered_tracks) {
      std::fill(first_density.begin(), first_density.end(), unselected);
      ForEachRowTrackObservation(
          source,
          track,
          [&](const uint32_t source_image,
              const uint32_t image,
              const float* xy) {
            const uint32_t frame = image_frame_indices[image];
            const float* dimensions =
                sensor_dimensions + 2 * image_sensor_indices[image];
            if (xy[0] < 0.0f || xy[0] >= dimensions[0] || xy[1] < 0.0f ||
                xy[1] >= dimensions[1]) {
              return;
            }
            const uint32_t cell_x =
                static_cast<uint32_t>(xy[0] * kSpatialGridSide / dimensions[0]);
            const uint32_t cell_y =
                static_cast<uint32_t>(xy[1] * kSpatialGridSide / dimensions[1]);
            const uint32_t cell = cell_y * kSpatialGridSide + cell_x;
            for (uint32_t membership = frame_section_offsets[frame];
                 membership < frame_section_offsets[frame + 1];
                 ++membership) {
              const uint32_t section = frame_sections[membership];
              uint32_t& count =
                  stratum_counts[(section * source.num_images + source_image) *
                                     kSpatialCellCount +
                                 cell];
              ++count;
              first_density[section] = std::min(first_density[section], count);
            }
          });
      const uint32_t row_point = source.row_point_indices[track];
      for (size_t section = 0; section < num_sections; ++section) {
        uint32_t& density =
            assignment.first_density[section * num_row_points + row_point];
        density = std::min(density, first_density[section]);
      }
    }
  }
  return assignment;
}

CasparRowPointDensities AssignCasparRowPointDensities(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const float* sensor_dimensions,
    const bool* active_frame_mask,
    const size_t num_frames,
    const uint32_t* point_observation_counts,
    const size_t num_row_points,
    const size_t minimum_track_length) {
  // A track is admitted at density d exactly when one of its strata holds
  // fewer than d earlier tracks, so the uncapped first density of the whole
  // row as a single section decides every density in one scan.
  CasparRowSectionDensities assignment =
      AssignCasparRowSectionDensities(sources,
                                      image_frame_indices,
                                      image_sensor_indices,
                                      sensor_dimensions,
                                      active_frame_mask,
                                      /*num_sections=*/1,
                                      num_frames,
                                      num_row_points,
                                      minimum_track_length);
  const uint32_t unselected = std::numeric_limits<uint32_t>::max();
  uint32_t maximum_density = 0;
  for (const uint32_t density : assignment.first_density) {
    if (density != unselected) {
      maximum_density = std::max(maximum_density, density);
    }
  }
  CasparRowPointDensities densities{
      std::move(assignment.first_density),
      std::vector<uint64_t>(maximum_density + 1),
      std::vector<uint64_t>(maximum_density + 1)};
  for (size_t row_point = 0; row_point < num_row_points; ++row_point) {
    const uint32_t density = densities.first_density[row_point];
    if (density != unselected) {
      ++densities.point_counts[density];
      densities.observation_counts[density] +=
          point_observation_counts[row_point];
    }
  }
  std::partial_sum(densities.point_counts.begin(),
                   densities.point_counts.end(),
                   densities.point_counts.begin());
  std::partial_sum(densities.observation_counts.begin(),
                   densities.observation_counts.end(),
                   densities.observation_counts.begin());
  return densities;
}

CasparRowTrackSelection SelectCasparRowPoints(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    const float* sensor_dimensions,
    const bool* active_frame_mask,
    const size_t num_row_points,
    const size_t minimum_track_length,
    const uint32_t tracks_per_spatial_cell) {
  const CasparRowTiers assignment =
      AssignCasparRowTiers(sources,
                           image_frame_indices,
                           image_sensor_indices,
                           sensor_dimensions,
                           active_frame_mask,
                           num_row_points,
                           minimum_track_length,
                           &tracks_per_spatial_cell,
                           1);
  CasparRowTrackSelection selection;
  selection.point_indices.resize(static_cast<size_t>(std::count(
      assignment.first_tier.begin(), assignment.first_tier.end(), uint8_t{0})));
  auto selected_point = selection.point_indices.begin();
  for (uint32_t row_point = 0; row_point < num_row_points; ++row_point) {
    if (assignment.first_tier[row_point] == 0) {
      *selected_point++ = row_point;
    }
  }
  selection.quota_truncated = assignment.quota_truncated;
  return selection;
}

CasparRowPointSelection InitializeRowPoints(
    const std::vector<CasparRowTrackSource>& sources,
    const std::vector<uint32_t>& source_track_offsets,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const uint32_t* row_point_indices,
    const size_t num_points) {
  CasparRowPointSelection selected{
      std::vector<uint32_t>(row_point_indices, row_point_indices + num_points),
      std::vector<float>(3 * num_points),
  };
  for (size_t point = 0; point < num_points; ++point) {
    const uint32_t row_point = row_point_indices[point];
    const uint32_t point_track_start = point_track_offsets[row_point];
    const uint32_t point_track_end = point_track_offsets[row_point + 1];
    Eigen::Vector3f point_sum = Eigen::Vector3f::Zero();
    ForEachRowPointTrack(sources,
                         source_track_offsets,
                         point_track_offsets,
                         point_track_indices,
                         row_point,
                         [&](const size_t,
                             const CasparRowTrackSource& source,
                             const uint32_t track) {
                           point_sum += RowSourcePoint(
                               source, source.source_point_indices[track]);
                         });
    Eigen::Map<Eigen::Vector3f>(selected.points.data() + 3 * point) =
        point_sum / static_cast<float>(point_track_end - point_track_start);
  }
  return selected;
}

}  // namespace colmap
