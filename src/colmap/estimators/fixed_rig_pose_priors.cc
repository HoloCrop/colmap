#include "colmap/estimators/fixed_rig_pose_priors.h"

#include "colmap/util/logging.h"

namespace colmap {

NodeHashMap<frame_t, const PosePrior*> FramePosePriors(
    const Reconstruction& reconstruction,
    const std::vector<PosePrior>& pose_priors) {
  FlatHashMap<data_t, frame_t> reference_data_frames;
  for (const auto& [frame_id, frame] : reconstruction.Frames()) {
    const sensor_t ref_sensor_id =
        reconstruction.Rig(frame.RigId()).RefSensorId();
    for (const data_t& data_id : frame.DataIds()) {
      if (data_id.sensor_id == ref_sensor_id) {
        reference_data_frames.emplace(data_id, frame_id);
      }
    }
  }

  NodeHashMap<frame_t, const PosePrior*> frame_pose_priors;
  for (const PosePrior& pose_prior : pose_priors) {
    const auto it = reference_data_frames.find(pose_prior.corr_data_id);
    if (it == reference_data_frames.end()) {
      continue;
    }
    THROW_CHECK(frame_pose_priors.emplace(it->second, &pose_prior).second)
        << "Duplicate pose prior for frame " << it->second;
  }
  return frame_pose_priors;
}

}  // namespace colmap
