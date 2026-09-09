#pragma once

#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/estimators/row_track_caspar.h"
#include "colmap/geometry/rigid3.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace colmap {

struct CasparRowBundleResult {
  std::vector<Rigid3d> rigs_from_world;
  std::vector<uint32_t> row_point_indices;
  std::vector<float> points;
  double sensor_from_rig_scale = 1.0;
  size_t observation_count = 0;
  double preparation_seconds = 0.0;
  double optimization_seconds = 0.0;
  std::shared_ptr<CasparBundleAdjustmentSummary> summary;
};

CasparRowBundleResult OptimizeRowCaspar(
    const std::vector<CasparRowTrackSource>& sources,
    const uint32_t* point_track_offsets,
    const uint32_t* point_track_indices,
    const uint32_t* point_observation_counts,
    const uint32_t* selected_row_point_indices,
    size_t num_selected_points,
    const std::vector<Rigid3d>& initial_rigs_from_world,
    const uint32_t* image_frame_indices,
    const uint32_t* image_sensor_indices,
    size_t num_images,
    const std::vector<Rigid3d>& sensors_from_rig,
    const float* sensor_calibrations,
    const uint32_t* prior_frame_indices,
    const float* prior_positions,
    const float* prior_sqrt_information,
    size_t num_priors,
    const CasparBundleAdjustmentOptions& options,
    bool refine_scale);

}  // namespace colmap
