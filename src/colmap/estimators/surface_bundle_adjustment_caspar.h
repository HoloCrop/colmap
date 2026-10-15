#pragma once

#include "colmap/estimators/bundle_adjustment_caspar.h"
#include "colmap/geometry/rigid3.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace colmap {

// Fixed-rig observations of one trajectory, in device memory (row-major).
struct CasparSurfaceObservations {
  const uint32_t* point_indices;   // (count,)
  const uint32_t* frame_indices;   // (count,)
  const uint32_t* sensor_indices;  // (count,)
  const float* pixels;             // (count, 2)
  size_t count;
};

// Surface factors in device memory (row-major). Each ties a point fixed in a
// rig frame to a fixed world position: the offset's length beyond `margin`
// goes through a Geman-McClure kernel of scale `robust_scale` (metres) and is
// multiplied by the factor's sqrt weight.
struct CasparSurfaceFactors {
  const uint32_t* frame_indices;  // (count,)
  const float* rig_points;        // (count, 3) in the frame's rig coordinates
  const float* positions;         // (count, 3)
  const float* sqrt_weights;      // (count,)
  size_t count;
  double robust_scale;
  double margin;
};

struct CasparSurfaceBundleResult {
  std::vector<Rigid3d> rigs_from_world;
  std::shared_ptr<CasparBundleAdjustmentSummary> summary;
};

// Fixed-rig bundle adjustment of one rig trajectory and its points, with the
// trajectory tied to a fixed surface (for example the other side of a crop
// row) through points it carries.
// Observations, calibration and rig layout are uploaded once; each Solve takes
// the current rig poses, points and surface factors, all point and factor data
// staying in device memory. The surface fixes the gauge, so no frame is
// anchored; calibration and the rig baseline scale are held fixed.
class CasparSurfaceBundleAdjuster {
 public:
  CasparSurfaceBundleAdjuster(
      const std::vector<Rigid3d>& initial_rigs_from_world,
      const std::vector<Rigid3d>& sensors_from_rig,
      const float* sensor_calibrations,  // host (sensors, 4): fx fy cx cy
      const CasparSurfaceObservations& observations,
      size_t num_points,
      size_t max_surface_factors,
      const CasparBundleAdjustmentOptions& options);
  ~CasparSurfaceBundleAdjuster();

  CasparSurfaceBundleAdjuster(const CasparSurfaceBundleAdjuster&) = delete;
  CasparSurfaceBundleAdjuster& operator=(const CasparSurfaceBundleAdjuster&) =
      delete;

  int DeviceId() const;

  // `points` (num_points, 3) in device memory is read and updated in place.
  CasparSurfaceBundleResult Solve(
      const std::vector<Rigid3d>& rigs_from_world,
      float* points,
      const CasparSurfaceFactors& factors);

 private:
  class Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace colmap
