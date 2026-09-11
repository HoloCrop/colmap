#include "colmap/estimators/bundle_adjustment_caspar.h"

#include "colmap/estimators/bundle_adjustment.h"
#include "colmap/estimators/bundle_adjustment_ceres.h"
#include "colmap/geometry/rigid3.h"
#include "colmap/scene/camera.h"
#include "colmap/scene/image.h"
#include "colmap/sensor/models.h"
#include "colmap/util/cuda.h"
#include "colmap/util/hash_containers.h"
#include "colmap/util/misc.h"
#ifdef CASPAR_ENABLED
#include "colmap/estimators/caspar/caspar_model_adapter.h"
#endif

#include <Eigen/Cholesky>

namespace colmap {
namespace {

enum class CasparSolverMode { STANDARD, RIG_SCHUR };

class CasparBundleAdjuster : public BundleAdjuster {
 public:
  CasparBundleAdjuster(BundleAdjustmentOptions options,
                       BundleAdjustmentConfig config,
                       Reconstruction& reconstruction,
                       CasparSolverMode solver_mode)
      : BundleAdjuster(options, config),
        solver_mode_(solver_mode),
        reconstruction_(reconstruction) {
    if (solver_mode_ == CasparSolverMode::RIG_SCHUR) {
      ValidateRigSchurContract();
      VLOG(2) << "Creating Caspar rig-Schur bundle adjuster";
    } else {
      VLOG(2) << "Creating Caspar bundle adjuster";
    }

    LogUnsupportedOptions();
    BuildObservationCounts();
    FixGauge();
    BuildFactors();
  }

 private:
  void LogUnsupportedOptions() const {
    if (options_.refine_rig_from_world &&
        options_.constant_rig_from_world_rotation) {
      LOG(ERROR)
          << "Caspar does not support constant_rig_from_world_rotation=true. "
             "The option will be ignored and eligible rig rotations will be "
             "refined.";
    }
  }

  void ValidateRigSchurContract() const {
#ifdef CASPAR_USE_DOUBLE
    LOG(FATAL_THROW)
        << "Caspar rig-Schur only supports the default float precision";
#endif
    if (options_.refine_focal_length || options_.refine_principal_point ||
        options_.refine_extra_params) {
      LOG(FATAL_THROW) << "Caspar rig-Schur requires fixed camera intrinsics";
    }
    if (options_.refine_sensor_from_rig) {
      LOG(FATAL_THROW)
          << "Caspar rig-Schur requires fixed sensor-from-rig poses";
    }
    if (!options_.refine_rig_from_world) {
      LOG(FATAL_THROW)
          << "Caspar rig-Schur requires variable rig-from-world poses";
    }
    if (!options_.refine_points3D) {
      LOG(FATAL_THROW) << "Caspar rig-Schur requires variable 3D points";
    }
    if (options_.constant_rig_from_world_rotation) {
      LOG(FATAL_THROW)
          << "Caspar rig-Schur does not support fixed rig-frame rotations";
    }
  }

  ICasparModelAdapter* GetAdapter(const CameraModelId model_id) {
    if (solver_mode_ == CasparSolverMode::RIG_SCHUR &&
        model_id != CameraModelId::kPinhole) {
      LOG(FATAL_THROW) << "Caspar rig-Schur only supports PINHOLE cameras";
    }

    auto it = adapters_.find(model_id);
    if (it != adapters_.end()) {
      return it->second.get();
    }

    auto adapter = CreateCasparAdapter(model_id);
    if (!adapter) {
      return nullptr;
    }

    auto* ptr = adapter.get();
    adapters_[model_id] = std::move(adapter);
    model_data_per_model_.emplace(model_id, ModelData{});
    calib_num_per_model_[model_id] = 0;
    return ptr;
  }

  void BuildObservationCounts() {
    for (const image_t image_id : config_.Images()) {
      const Image& image = reconstruction_.Image(image_id);
      const Camera& camera = *image.CameraPtr();
      if (!GetAdapter(camera.model_id)) {
        LOG(WARNING) << "Skipping image " << image_id
                     << " with unsupported camera model: "
                     << camera.ModelName();
        continue;
      }
      size_t num_observations = 0;
      for (const Point2D& point2D : image.Points2D()) {
        if (!point2D.HasPoint3D() ||
            config_.IsIgnoredPoint(point2D.point3D_id) ||
            !HasSufficientTrackLength(point2D.point3D_id)) {
          continue;
        }
        point3D_num_observations_[point2D.point3D_id]++;
        ++num_observations;
      }
      if (solver_mode_ == CasparSolverMode::RIG_SCHUR) {
        rig_schur_image_counts_[image_id].variable_points = num_observations;
      }
    }
    for (const auto point3D_id : config_.VariablePoints()) {
      CountExternalObservations(point3D_id);
    }
    for (const auto point3D_id : config_.ConstantPoints()) {
      CountExternalObservations(point3D_id);
    }
  }

  void CountExternalObservations(const point3D_t point3D_id) {
    if (!HasSufficientTrackLength(point3D_id)) {
      return;
    }
    const Point3D& point3D = reconstruction_.Point3D(point3D_id);
    for (const auto& track_el : point3D.track.Elements()) {
      if (!config_.HasImage(track_el.image_id)) {
        Image& image = reconstruction_.Image(track_el.image_id);
        Camera& camera = *image.CameraPtr();
        if (GetAdapter(camera.model_id)) {
          point3D_num_observations_[point3D_id]++;
        }
      }
    }
  }

  void BuildFactors() {
    CreateCalibrationNodes();
    CreatePoseNodes();
    CreatePointNodes();
    if (solver_mode_ == CasparSolverMode::RIG_SCHUR) {
      ReserveRigSchurFactors();
    }
    AddFactors();
    AddExternalFactors();
  }

  void CreateCalibrationNodes() {
    std::vector<camera_t> sorted_camera_ids;
    sorted_camera_ids.reserve(config_.Images().size());
    for (const image_t image_id : config_.Images()) {
      sorted_camera_ids.push_back(reconstruction_.Image(image_id).CameraId());
    }
    std::sort(sorted_camera_ids.begin(), sorted_camera_ids.end());
    sorted_camera_ids.erase(
        std::unique(sorted_camera_ids.begin(), sorted_camera_ids.end()),
        sorted_camera_ids.end());

    for (const camera_t camera_id : sorted_camera_ids) {
      const Camera& camera = reconstruction_.Camera(camera_id);
      if (!GetAdapter(camera.model_id)) {
        continue;
      }
      GetOrCreateCalibration(camera_id, camera);
    }
  }

  void CreatePoseNodes() {
    std::map<frame_t, CameraModelId> frame_to_model;
    for (const image_t image_id : config_.Images()) {
      const Image& image = reconstruction_.Image(image_id);
      const Camera& camera = reconstruction_.Camera(image.CameraId());
      frame_to_model.emplace(image.FrameId(), camera.model_id);
    }
    for (const auto& [frame_id, model_id] : frame_to_model) {
      GetOrCreatePose(frame_id, model_id);
    }
  }

  void CreatePointNodes() {
    std::vector<point3D_t> sorted_point3D_ids;
    sorted_point3D_ids.reserve(point3D_num_observations_.size());
    for (const auto& [point_id, _] : point3D_num_observations_) {
      sorted_point3D_ids.push_back(point_id);
    }
    std::sort(sorted_point3D_ids.begin(), sorted_point3D_ids.end());
    point3D_id_to_idx_.reserve(sorted_point3D_ids.size());
    point3D_idx_to_id_.reserve(sorted_point3D_ids.size());
    point3D_data_.reserve(sorted_point3D_ids.size() * 3);
    for (const point3D_t point_id : sorted_point3D_ids) {
      GetOrCreatePoint(point_id, reconstruction_.Point3D(point_id));
      // The gauge is now fixed. Correct the image counts by walking only
      // constant-point tracks, rather than rescanning every image observation.
      if (solver_mode_ == CasparSolverMode::RIG_SCHUR &&
          !IsPointVariable(point_id)) {
        for (const auto& element :
             reconstruction_.Point3D(point_id).track.Elements()) {
          if (config_.HasImage(element.image_id)) {
            auto& counts = rig_schur_image_counts_.at(element.image_id);
            --counts.variable_points;
            ++counts.fixed_points;
          }
        }
      }
    }
  }

  void ReserveRigSchurFactors() {
    size_t main_count = 0;
    size_t fixed_pose_count = 0;
    size_t fixed_point_count = 0;
    for (const auto& [image_id, counts] : rig_schur_image_counts_) {
      if (IsPoseVariable(reconstruction_.Image(image_id).FrameId())) {
        main_count += counts.variable_points;
        fixed_point_count += counts.fixed_points;
      } else {
        fixed_pose_count += counts.variable_points;
      }
    }
    // External observations have fixed poses and intrinsics. Fully constant
    // factors are omitted by AddFactorCore, including fixed points here.
    for (const point3D_t point_id : config_.VariablePoints()) {
      if (!HasSufficientTrackLength(point_id) || !IsPointVariable(point_id)) {
        continue;
      }
      for (const auto& element :
           reconstruction_.Point3D(point_id).track.Elements()) {
        fixed_pose_count += !config_.HasImage(element.image_id);
      }
    }
    if (main_count == 0 && fixed_pose_count == 0 && fixed_point_count == 0) {
      return;
    }
    auto& variants = model_data_per_model_.at(CameraModelId::kPinhole).variants;
    const auto reserve = [&](FactorVariant variant, size_t count,
                             bool pose_variable, bool point_variable) {
      auto& data = variants[static_cast<size_t>(variant)];
      data.pixels.reserve(2 * count);
      data.sensor_from_rig_data.reserve(7 * count);
      data.const_focal_and_extra.reserve(2 * count);
      data.const_principal_point.reserve(2 * count);
      if (pose_variable) data.pose_indices.reserve(count);
      else data.const_poses.reserve(7 * count);
      if (point_variable) data.point_indices.reserve(count);
      else data.const_points.reserve(3 * count);
    };
    reserve(FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT,
            main_count, true, true);
    reserve(FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT,
            fixed_pose_count, false, true);
    reserve(FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT,
            fixed_point_count, true, false);
  }

  void AddFactors() {
    // Sort camera-first so all factors for the same calibration are contiguous.
    // This improves float32 numerical quality in the GPU gradient summation.
    std::vector<std::pair<camera_t, image_t>> sorted_images;
    sorted_images.reserve(config_.Images().size());
    for (const image_t image_id : config_.Images()) {
      sorted_images.emplace_back(reconstruction_.Image(image_id).CameraId(),
                                 image_id);
    }
    std::sort(sorted_images.begin(), sorted_images.end());

    // Cache per-camera values across images sharing the same camera. These are
    // recomputed only when the camera changes (images are sorted camera-first).
    camera_t prev_camera_id = static_cast<camera_t>(-1);
    ICasparModelAdapter* adapter = nullptr;
    const Camera* camera_ptr = nullptr;
    bool focal_and_extra = false;
    bool principal_point_var = false;
    size_t calib_idx = 0;

    for (const auto& [camera_id, image_id] : sorted_images) {
      const Image& image = reconstruction_.Image(image_id);

      if (camera_id != prev_camera_id) {
        camera_ptr = &reconstruction_.Camera(camera_id);
        adapter = GetAdapter(camera_ptr->model_id);
        if (adapter) {
          if (options_.refine_focal_length != options_.refine_extra_params &&
              !config_.HasConstantCamIntrinsics(camera_id) &&
              !cameras_from_outside_config_.count(camera_id)) {
            LOG(FATAL_THROW)
                << "Camera " << camera_id
                << ": refine_focal_length != refine_extra_params is not "
                   "supported by CASPAR's merged focal_and_extra block.";
          }
          focal_and_extra = IsFocalAndExtraVariable(camera_id);
          principal_point_var = IsPrincipalPointVariable(camera_id);
          calib_idx = GetOrCreateCalibration(camera_id, *camera_ptr);
        }
        prev_camera_id = camera_id;
      }
      if (!adapter) continue;

      if (options_.refine_sensor_from_rig) {
        const Frame& frame = *image.FramePtr();
        if (frame.HasRigPtr() && !image.IsRefInFrame()) {
          LOG(FATAL_THROW)
              << "Camera " << camera_id
              << ": refine_sensor_from_rig=true is not supported by CASPAR. "
                 "Set refine_sensor_from_rig=false or use the Ceres BA.";
        }
      }
      const bool pose_var = IsPoseVariable(image.FrameId());

      for (const Point2D& point2D : image.Points2D()) {
        if (!point2D.HasPoint3D() ||
            config_.IsIgnoredPoint(point2D.point3D_id) ||
            !point3D_id_to_idx_.count(point2D.point3D_id)) {
          continue;
        }
        AddFactorCore(image,
                      *camera_ptr,
                      point2D,
                      reconstruction_.Point3D(point2D.point3D_id),
                      pose_var,
                      focal_and_extra,
                      principal_point_var,
                      calib_idx,
                      *adapter);
      }
    }
  }

  void AddExternalFactors() {
    for (const auto point3D_id : config_.VariablePoints()) {
      AddFactorsForExternalObservations(point3D_id);
    }
    for (const auto point3D_id : config_.ConstantPoints()) {
      AddFactorsForExternalObservations(point3D_id);
    }
  }

  void AddFactorsForExternalObservations(const point3D_t point3D_id) {
    THROW_CHECK(!config_.IsIgnoredPoint(point3D_id));
    if (!HasSufficientTrackLength(point3D_id)) {
      return;
    }
    Point3D& point3D = reconstruction_.Point3D(point3D_id);
    GetOrCreatePoint(point3D_id, point3D);

    for (const auto& track_el : point3D.track.Elements()) {
      if (config_.HasImage(track_el.image_id)) {
        continue;
      }
      Image& image = reconstruction_.Image(track_el.image_id);
      Camera& camera = *image.CameraPtr();
      ICasparModelAdapter* adapter = GetAdapter(camera.model_id);
      if (!adapter) {
        LOG(WARNING) << "Skipping external observation with unsupported "
                        "camera model: "
                     << camera.ModelName();
        continue;
      }
      // Mark frame and camera as external so that IsPoseVariable and
      // IsFocalAndExtraVariable return false for all external
      // observations.
      frames_from_outside_config_.insert(image.FrameId());
      cameras_from_outside_config_.insert(camera.camera_id);
      if (options_.refine_sensor_from_rig) {
        const Frame& frame = *image.FramePtr();
        if (frame.HasRigPtr() && !image.IsRefInFrame()) {
          LOG(FATAL_THROW)
              << "Camera " << camera.camera_id
              << ": refine_sensor_from_rig=true is not supported by CASPAR. "
                 "Set refine_sensor_from_rig=false or use the Ceres BA.";
        }
      }
      const Point2D& point2D = image.Point2D(track_el.point2D_idx);
      const size_t calib_idx = GetOrCreateCalibration(camera.camera_id, camera);
      AddFactorCore(image,
                    camera,
                    point2D,
                    point3D,
                    /*pose_var=*/false,
                    /*focal_and_extra=*/false,
                    /*principal_point_var=*/false,
                    calib_idx,
                    *adapter);
    }
  }

  void AddFactorCore(const Image& image,
                     const Camera& camera,
                     const Point2D& point2D,
                     const Point3D& point3D,
                     bool pose_var,
                     bool focal_and_extra,
                     bool principal_point_var,
                     size_t calib_idx,
                     ICasparModelAdapter& adapter) {
    const bool point_var = IsPointVariable(point2D.point3D_id);

    // Skip fully-constant observations, there's nothing to optimize
    if (!pose_var && !focal_and_extra && !principal_point_var && !point_var) {
      return;
    }

    ModelData& md = model_data_per_model_.at(camera.model_id);

    // 4-bit key: bit3=pose_var, bit2=fae_var, bit1=pp_var, bit0=pt_var.
    // Entry 0 (all fixed) is unreachable
    static constexpr FactorVariant kVariantTable[16] = {
        /* 0000 */ FactorVariant::BASE,  // unreachable
                                         /* 0001 */
        FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT,
        /* 0010 */
        FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_POINT,
        /* 0011 */ FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA,
        /* 0100 */ FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT_FIXED_POINT,
        /* 0101 */ FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT,
        /* 0110 */ FactorVariant::FIXED_POSE_FIXED_POINT,
        /* 0111 */ FactorVariant::FIXED_POSE,
        /* 1000 */
        FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT,
        /* 1001 */
        FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT,
        /* 1010 */ FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_POINT,
        /* 1011 */ FactorVariant::FIXED_FOCAL_AND_EXTRA,
        /* 1100 */ FactorVariant::FIXED_PRINCIPAL_POINT_FIXED_POINT,
        /* 1101 */ FactorVariant::FIXED_PRINCIPAL_POINT,
        /* 1110 */ FactorVariant::FIXED_POINT,
        /* 1111 */ FactorVariant::BASE,
    };
    const FactorVariant v =
        kVariantTable[(static_cast<int>(pose_var) << 3) |
                      (static_cast<int>(focal_and_extra) << 2) |
                      (static_cast<int>(principal_point_var) << 1) |
                      static_cast<int>(point_var)];

    VariantData& vd = md.variants[static_cast<int>(v)];

    AppendPose(vd.sensor_from_rig_data, GetSensorFromRig(image));

    if (pose_var) {
      vd.pose_indices.push_back(
          GetOrCreatePose(image.FrameId(), camera.model_id));
    } else {
      AppendPose(vd.const_poses,
                 reconstruction_.Frame(image.FrameId()).RigFromWorld());
    }

    if (focal_and_extra) {
      vd.focal_and_extra_indices.push_back(calib_idx);
    } else {
      const size_t fs = adapter.FocalAndExtraSize();
      const auto* src = md.focal_and_extra_data.data() + calib_idx * fs;
      vd.const_focal_and_extra.insert(
          vd.const_focal_and_extra.end(), src, src + fs);
    }

    if (principal_point_var) {
      vd.principal_point_indices.push_back(calib_idx);
    } else {
      const size_t ps = adapter.PrincipalPointSize();
      const auto* src = md.principal_point_data.data() + calib_idx * ps;
      vd.const_principal_point.insert(
          vd.const_principal_point.end(), src, src + ps);
    }

    if (point_var) {
      vd.point_indices.push_back(GetOrCreatePoint(point2D.point3D_id, point3D));
    } else {
      AppendPoint(vd.const_points, point3D);
    }

    vd.pixels.push_back(point2D.xy.x());
    vd.pixels.push_back(point2D.xy.y());
    ++vd.num_factors;
  }

  static void AppendPose(std::vector<StorageType>& out, const Rigid3d& pose) {
    out.push_back(pose.rotation().x());
    out.push_back(pose.rotation().y());
    out.push_back(pose.rotation().z());
    out.push_back(pose.rotation().w());
    out.push_back(pose.translation().x());
    out.push_back(pose.translation().y());
    out.push_back(pose.translation().z());
  }

  static void AppendPoint(std::vector<StorageType>& out, const Point3D& pt) {
    out.push_back(pt.xyz.x());
    out.push_back(pt.xyz.y());
    out.push_back(pt.xyz.z());
  }

  size_t GetOrCreatePoint(const point3D_t point_id, const Point3D& point) {
    auto [it, inserted] = point3D_id_to_idx_.try_emplace(point_id, num_points_);
    if (inserted) {
      point3D_idx_to_id_.push_back(point_id);
      point3D_data_.push_back(point.xyz.x());
      point3D_data_.push_back(point.xyz.y());
      point3D_data_.push_back(point.xyz.z());
      num_points_++;
    }
    return it->second;
  }

  size_t GetOrCreatePose(const frame_t frame_id, const CameraModelId model_id) {
    size_t& n = num_poses_per_model_[model_id];
    auto [it, inserted] =
        frame_to_pose_index_per_model_[model_id].try_emplace(frame_id, n);
    if (inserted) {
      pose_index_to_frame_per_model_[model_id][n] = frame_id;
      const Rigid3d& pose = reconstruction_.Frame(frame_id).RigFromWorld();
      auto& data = pose_data_per_model_[model_id];
      data.push_back(pose.rotation().x());
      data.push_back(pose.rotation().y());
      data.push_back(pose.rotation().z());
      data.push_back(pose.rotation().w());
      data.push_back(pose.translation().x());
      data.push_back(pose.translation().y());
      data.push_back(pose.translation().z());
      n++;
    }
    return it->second;
  }

  // Returns the sensor_from_rig transform for a camera. Identity for ref
  // sensors and single-camera datasets; actual transform for non-ref rigs.
  Rigid3d GetSensorFromRig(const Image& image) {
    const Frame& frame = *image.FramePtr();
    if (frame.HasRigPtr() && !image.IsRefInFrame()) {
      return frame.RigPtr()->SensorFromRig(image.DataId().sensor_id);
    }
    return Rigid3d{};
  }

  // Calib indices are per-model: index 0 for SimpleRadial is unrelated to
  // index 0 for Pinhole.
  size_t GetOrCreateCalibration(const camera_t camera_id,
                                const Camera& camera) {
    auto [it, inserted] = camera_to_calib_index_.try_emplace(camera_id, 0);
    if (inserted) {
      ICasparModelAdapter* adapter = GetAdapter(camera.model_id);
      size_t& model_calib_count = calib_num_per_model_[camera.model_id];
      it->second = model_calib_count;
      calib_index_to_camera_[{camera.model_id, model_calib_count}] = camera_id;
      ModelData& md = model_data_per_model_.at(camera.model_id);
      adapter->ExtractFocalAndExtra(camera, md.focal_and_extra_data);
      adapter->ExtractPrincipalPoint(camera, md.principal_point_data);
      ++model_calib_count;
    }
    return it->second;
  }

  bool IsPoseVariable(const frame_t frame_id) const {
    return options_.refine_rig_from_world &&
           !config_.HasConstantRigFromWorldPose(frame_id) &&
           !frames_from_outside_config_.count(frame_id) &&
           !gauge_fixed_frames_.count(frame_id);
  }

  // Both focal and extra_params must be refined together (merged block).
  // If they disagree, observations are skipped. See AddFactorForObservation.
  bool IsFocalAndExtraVariable(const camera_t camera_id) const {
    return options_.refine_focal_length && options_.refine_extra_params &&
           !config_.HasConstantCamIntrinsics(camera_id) &&
           !cameras_from_outside_config_.count(camera_id);
  }

  bool IsPrincipalPointVariable(const camera_t camera_id) const {
    return options_.refine_principal_point &&
           !config_.HasConstantCamIntrinsics(camera_id) &&
           !cameras_from_outside_config_.count(camera_id);
  }

  bool AreIntrinsicsVariable(const camera_t camera_id) const {
    return IsFocalAndExtraVariable(camera_id) ||
           IsPrincipalPointVariable(camera_id);
  }

  bool IsPointVariable(const point3D_t point3D_id) const {
    if (!options_.refine_points3D || config_.HasConstantPoint(point3D_id) ||
        gauge_fixed_points_.count(point3D_id)) {
      return false;
    }
    const auto it = point3D_num_observations_.find(point3D_id);
    return it != point3D_num_observations_.end() &&
           reconstruction_.Point3D(point3D_id).track.Length() <= it->second;
  }

  bool HasSufficientTrackLength(const point3D_t point3D_id) const {
    return options_.min_track_length <= 0 ||
           static_cast<int>(
               reconstruction_.Point3D(point3D_id).track.Length()) >=
               options_.min_track_length;
  }

  void FixGauge() {
    switch (config_.FixedGauge()) {
      case BundleAdjustmentGauge::UNSPECIFIED:
        break;
      case BundleAdjustmentGauge::TWO_CAMS_FROM_WORLD:
        FixGaugeWithOneFrameFromWorld();
        break;
      case BundleAdjustmentGauge::THREE_POINTS:
        FixGaugeWithThreePoints();
        break;
      default:
        LOG(FATAL_THROW) << "Unknown BundleAdjustmentGauge";
    }
  }

  // Partial two-view gauge fix: fixes one pose only. Caspar can express the
  // second camera's 1-DOF translation manifold, but the gain is minimal and
  // the extra shared-memory cost is not worth it, so scale is left as the one
  // unfixed gauge DOF.
  void FixGaugeWithOneFrameFromWorld() {
    if (!options_.refine_rig_from_world) {
      return;
    }

    // Sort image IDs for deterministic selection (matches Ceres BA behavior).
    std::vector<image_t> sorted_image_ids(config_.Images().begin(),
                                          config_.Images().end());
    std::sort(sorted_image_ids.begin(), sorted_image_ids.end());

    for (const image_t image_id : sorted_image_ids) {
      const Image& image = reconstruction_.Image(image_id);
      if (config_.HasConstantRigFromWorldPose(image.FrameId())) {
        VLOG(1) << "Gauge fix: frame " << image.FrameId()
                << " already constant, skipping TWO_CAMS_FROM_WORLD fix";
        return;
      }
    }

    // Require a ref-sensor image so fixing the frame moves factors into
    // fixed-pose variants rather than being silently skipped.
    for (const image_t image_id : sorted_image_ids) {
      const Image& image = reconstruction_.Image(image_id);
      if (image.IsRefInFrame()) {
        gauge_fixed_frames_.insert(image.FrameId());
        VLOG(1) << "Gauge fix: fixed frame " << image.FrameId() << " (image "
                << image_id << ") for TWO_CAMS_FROM_WORLD";
        return;
      }
    }

    LOG(WARNING) << "Caspar TWO_CAMS_FROM_WORLD gauge fix: no ref-sensor "
                    "frame found, gauge left unfixed.";
  }

  // Three-point gauge fix: mirrors the Ceres BA equivalent but promotes points
  // into gauge_fixed_points_ instead of calling SetParameterBlockConstant.
  void FixGaugeWithThreePoints() {
    Eigen::Index num_fixed = 0;
    Eigen::Matrix3d fixed_pts = Eigen::Matrix3d::Zero();

    auto maybe_add = [&](const Eigen::Vector3d& xyz) -> bool {
      if (num_fixed >= 3) return false;
      fixed_pts.col(num_fixed) = xyz;
      if (fixed_pts.colPivHouseholderQr().rank() > num_fixed) {
        ++num_fixed;
        return true;
      }
      fixed_pts.col(num_fixed).setZero();
      return false;
    };

    // First pass: count already-constant points.
    for (const auto& [point3D_id, _] : point3D_num_observations_) {
      if (!config_.HasConstantPoint(point3D_id)) continue;
      const Point3D& pt = reconstruction_.Point3D(point3D_id);
      if (maybe_add(pt.xyz) && num_fixed >= 3) {
        VLOG(1) << "Gauge fix: 3 linearly independent constant points found, "
                   "THREE_POINTS gauge fixed";
        return;
      }
    }

    // Second pass: promote variable points to gauge-fixed.
    for (const auto& [point3D_id, _] : point3D_num_observations_) {
      if (!IsPointVariable(point3D_id)) continue;
      const Point3D& pt = reconstruction_.Point3D(point3D_id);
      if (maybe_add(pt.xyz)) {
        gauge_fixed_points_.insert(point3D_id);
        if (num_fixed >= 3) {
          VLOG(1) << "Gauge fix: fixed " << gauge_fixed_points_.size()
                  << " points for THREE_POINTS gauge";
          return;
        }
      }
    }

    LOG(WARNING) << "Caspar THREE_POINTS gauge fix: only " << num_fixed
                 << " of 3 linearly independent points found.";
  }

  void SetupSolverData(caspar::GraphSolver& solver) {
    VLOG(2) << "=== CASPAR SOLVER SETUP ===";
    VLOG(2) << "  Points: " << num_points_;

    if (num_points_ > 0) {
      solver.SetPointNodesFromStackedHost(point3D_data_.data(), 0, num_points_);
    }

    for (const auto& [model_id, adapter_ptr] : adapters_) {
      const ModelData& md = model_data_per_model_.at(model_id);
      const size_t n_calib = calib_num_per_model_.at(model_id);
      const size_t n_poses = num_poses_per_model_.count(model_id)
                                 ? num_poses_per_model_.at(model_id)
                                 : 0;

      VLOG(2) << "  Poses (" << static_cast<int>(model_id) << "): " << n_poses;

      if (n_poses > 0) {
        adapter_ptr->SetPoseNodes(
            solver, pose_data_per_model_.at(model_id).data(), n_poses);
      }

      if (n_calib > 0) {
        adapter_ptr->SetFocalAndExtraNodes(
            solver,
            const_cast<StorageType*>(md.focal_and_extra_data.data()),
            n_calib);
        adapter_ptr->SetPrincipalPointNodes(
            solver,
            const_cast<StorageType*>(md.principal_point_data.data()),
            n_calib);

        // Set merged Calib nodes when both intrinsic groups are tunable.
        const bool has_merged =
            md.variants[static_cast<int>(FactorVariant::BASE)].num_factors >
                0 ||
            md.variants[static_cast<int>(FactorVariant::FIXED_POSE)]
                    .num_factors > 0 ||
            md.variants[static_cast<int>(FactorVariant::FIXED_POINT)]
                    .num_factors > 0 ||
            md.variants[static_cast<int>(FactorVariant::FIXED_POSE_FIXED_POINT)]
                    .num_factors > 0;
        if (has_merged) {
          const size_t fae_size = adapter_ptr->FocalAndExtraSize();
          const size_t pp_size = adapter_ptr->PrincipalPointSize();
          const size_t cal_size = adapter_ptr->CalibSize();
          std::vector<StorageType> calib_data(n_calib * cal_size);
          for (size_t i = 0; i < n_calib; ++i) {
            for (size_t j = 0; j < fae_size; ++j) {
              calib_data[i * cal_size + j] =
                  md.focal_and_extra_data[i * fae_size + j];
            }
            for (size_t j = 0; j < pp_size; ++j) {
              calib_data[i * cal_size + fae_size + j] =
                  md.principal_point_data[i * pp_size + j];
            }
          }
          if (n_calib > 0) {
            VLOG(2) << "  SetCalibNodes [cam 0, model "
                    << static_cast<int>(model_id) << "]: [" << calib_data[0]
                    << ", " << calib_data[1] << ", " << calib_data[2] << ", "
                    << calib_data[3] << "]";
          }
          adapter_ptr->SetCalibNodes(solver, calib_data.data(), n_calib);
        }
      }
      for (int v = 0; v < CASPAR_NUM_VARIANTS; ++v) {
        if (md.variants[v].num_factors > 0) {
          adapter_ptr->SetVariantFactors(
              solver, static_cast<FactorVariant>(v), md.variants[v]);
        }
      }
    }
    solver.finish_indices();
  }

  void ReadSolverResults(caspar::GraphSolver& solver) {
    if (num_points_ > 0) {
      solver.GetPointNodesToStackedHost(point3D_data_.data(), 0, num_points_);
    }

    for (const auto& [model_id, adapter_ptr] : adapters_) {
      const size_t n_poses = num_poses_per_model_.count(model_id)
                                 ? num_poses_per_model_.at(model_id)
                                 : 0;
      if (n_poses > 0) {
        adapter_ptr->GetPoseNodes(
            solver, pose_data_per_model_.at(model_id).data(), n_poses);
      }
    }

    for (const auto& [model_id, adapter_ptr] : adapters_) {
      ModelData& md = model_data_per_model_.at(model_id);
      const size_t n_calib = calib_num_per_model_.at(model_id);
      if (n_calib > 0) {
        adapter_ptr->GetFocalAndExtraNodes(
            solver, md.focal_and_extra_data.data(), n_calib);
        adapter_ptr->GetPrincipalPointNodes(
            solver, md.principal_point_data.data(), n_calib);

        // Split the merged Calib node back into focal_and_extra_data and
        // principal_point_data, overwriting the stale split-pool values above.
        const bool has_merged =
            md.variants[static_cast<int>(FactorVariant::BASE)].num_factors >
                0 ||
            md.variants[static_cast<int>(FactorVariant::FIXED_POSE)]
                    .num_factors > 0 ||
            md.variants[static_cast<int>(FactorVariant::FIXED_POINT)]
                    .num_factors > 0 ||
            md.variants[static_cast<int>(FactorVariant::FIXED_POSE_FIXED_POINT)]
                    .num_factors > 0;
        if (has_merged) {
          const size_t fae_size = adapter_ptr->FocalAndExtraSize();
          const size_t pp_size = adapter_ptr->PrincipalPointSize();
          const size_t cal_size = adapter_ptr->CalibSize();
          std::vector<StorageType> calib_data(n_calib * cal_size);
          adapter_ptr->GetCalibNodes(solver, calib_data.data(), n_calib);
          if (n_calib > 0) {
            VLOG(2) << "  GetCalibNodes [cam 0, model "
                    << static_cast<int>(model_id) << "]: [" << calib_data[0]
                    << ", " << calib_data[1] << ", " << calib_data[2] << ", "
                    << calib_data[3] << "]";
          }
          for (size_t i = 0; i < n_calib; ++i) {
            for (size_t j = 0; j < fae_size; ++j) {
              md.focal_and_extra_data[i * fae_size + j] =
                  calib_data[i * cal_size + j];
            }
            for (size_t j = 0; j < pp_size; ++j) {
              md.principal_point_data[i * pp_size + j] =
                  calib_data[i * cal_size + fae_size + j];
            }
          }
          if (n_calib > 0) {
            VLOG(2) << "  After split-back [cam 0]: fae=["
                    << md.focal_and_extra_data[0] << ", "
                    << md.focal_and_extra_data[1] << "] pp=["
                    << md.principal_point_data[0] << ", "
                    << md.principal_point_data[1] << "]";
          }
        }
      }
    }
  }

  void WriteCalibsToReconstruction() {
    for (const auto& [camera_id, calib_idx] : camera_to_calib_index_) {
      if (!AreIntrinsicsVariable(camera_id)) {
        continue;
      }
      Camera& camera = reconstruction_.Camera(camera_id);
      ICasparModelAdapter* adapter = GetAdapter(camera.model_id);
      const ModelData& md = model_data_per_model_.at(camera.model_id);
      const std::string params_before = camera.ParamsToString();
      if (IsFocalAndExtraVariable(camera_id)) {
        adapter->WriteFocalAndExtra(
            camera, md.focal_and_extra_data.data(), calib_idx);
      }
      if (IsPrincipalPointVariable(camera_id)) {
        adapter->WritePrincipalPoint(
            camera, md.principal_point_data.data(), calib_idx);
      }
      THROW_CHECK(camera.VerifyParams());
      VLOG(1) << "Camera " << camera_id << " (" << camera.ModelName() << ")"
              << " params: [" << params_before << "] -> ["
              << camera.ParamsToString() << "]";
    }
  }

  void WriteResultsToReconstruction() {
    for (size_t idx = 0; idx < point3D_idx_to_id_.size(); ++idx) {
      const point3D_t point_id = point3D_idx_to_id_[idx];
      // Points with external observations are non-variable but have solver
      // nodes holding float copies; skip to avoid writing back stale values.
      if (!IsPointVariable(point_id)) {
        continue;
      }
      Point3D& point = reconstruction_.Point3D(point_id);
      point.xyz.x() = point3D_data_[idx * 3 + 0];
      point.xyz.y() = point3D_data_[idx * 3 + 1];
      point.xyz.z() = point3D_data_[idx * 3 + 2];
    }

    for (const auto& [model_id, idx_to_frame] :
         pose_index_to_frame_per_model_) {
      const auto& data = pose_data_per_model_.at(model_id);
      for (const auto& [idx, frame_id] : idx_to_frame) {
        if (!IsPoseVariable(frame_id)) {
          continue;
        }
        Rigid3d& pose = reconstruction_.Frame(frame_id).RigFromWorld();
        pose.rotation().x() = data[idx * 7 + 0];
        pose.rotation().y() = data[idx * 7 + 1];
        pose.rotation().z() = data[idx * 7 + 2];
        pose.rotation().w() = data[idx * 7 + 3];
        pose.translation().x() = data[idx * 7 + 4];
        pose.translation().y() = data[idx * 7 + 5];
        pose.translation().z() = data[idx * 7 + 6];
        pose.rotation().normalize();
      }
    }

    WriteCalibsToReconstruction();
  }

  static const char* FactorVariantName(FactorVariant v) {
    switch (v) {
      case FactorVariant::BASE:
        return "BASE";
      case FactorVariant::FIXED_POSE:
        return "FIXED_POSE";
      case FactorVariant::FIXED_FOCAL_AND_EXTRA:
        return "FIXED_FAE";
      case FactorVariant::FIXED_PRINCIPAL_POINT:
        return "FIXED_PP";
      case FactorVariant::FIXED_POINT:
        return "FIXED_POINT";
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA:
        return "FIXED_POSE_FAE";
      case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT:
        return "FIXED_POSE_PP";
      case FactorVariant::FIXED_POSE_FIXED_POINT:
        return "FIXED_POSE_POINT";
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
        return "FIXED_FAE_PP";
      case FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
        return "FIXED_FAE_POINT";
      case FactorVariant::FIXED_PRINCIPAL_POINT_FIXED_POINT:
        return "FIXED_PP_POINT";
      case FactorVariant::
          FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT:
        return "FIXED_POSE_FAE_PP";
      case FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_POINT:
        return "FIXED_POSE_FAE_POINT";
      case FactorVariant::FIXED_POSE_FIXED_PRINCIPAL_POINT_FIXED_POINT:
        return "FIXED_POSE_PP_POINT";
      case FactorVariant::
          FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT:
        return "FIXED_FAE_PP_POINT";
      default:
        return "UNKNOWN";
    }
  }

  void LogFactorDistribution() const {
    VLOG(1) << "=== Caspar factor distribution ===";
    VLOG(1) << "  Points: " << num_points_ << "  Frames: " << TotalPoses();
    for (const auto& [model_id, md] : model_data_per_model_) {
      for (int v = 0; v < CASPAR_NUM_VARIANTS; ++v) {
        if (md.variants[v].num_factors == 0) {
          continue;
        }
        VLOG(1) << "  model=" << static_cast<int>(model_id) << " variant="
                << FactorVariantName(static_cast<FactorVariant>(v))
                << " factors=" << md.variants[v].num_factors;
      }
    }
    VLOG(1) << "  Gauge-fixed frames: " << gauge_fixed_frames_.size()
            << "  Gauge-fixed points: " << gauge_fixed_points_.size();
    VLOG(1) << "  refine_focal_length=" << options_.refine_focal_length
            << " refine_extra_params=" << options_.refine_extra_params
            << " refine_pp=" << options_.refine_principal_point
            << " refine_pose=" << options_.refine_rig_from_world;
  }

  size_t ComputeTotalResiduals() const {
    size_t total = 0;
    for (const auto& [model_id, md] : model_data_per_model_) {
      for (int v = 0; v < CASPAR_NUM_VARIANTS; ++v) {
        total += md.variants[v].num_factors;
      }
    }
    return 2 * total;
  }

  CasparSolverSizing BuildSizing() const {
    CasparSolverSizing sz;
    sz.num_points = num_points_;
    if (auto it = num_poses_per_model_.find(CameraModelId::kSimpleRadial);
        it != num_poses_per_model_.end()) {
      sz.num_simple_radial_poses = it->second;
    }
    if (auto it = num_poses_per_model_.find(CameraModelId::kPinhole);
        it != num_poses_per_model_.end()) {
      sz.num_pinhole_poses = it->second;
    }
    auto get_md = [&](CameraModelId id) -> const ModelData* {
      auto it = model_data_per_model_.find(id);
      return it != model_data_per_model_.end() ? &it->second : nullptr;
    };
    auto get_n = [&](CameraModelId id) -> size_t {
      auto it = calib_num_per_model_.find(id);
      return it != calib_num_per_model_.end() ? it->second : 0;
    };

    if (const ModelData* md = get_md(CameraModelId::kSimpleRadial)) {
      sz.num_simple_radial_calibs = get_n(CameraModelId::kSimpleRadial);
      adapters_.at(CameraModelId::kSimpleRadial)
          ->FillSizing(sz, *md, sz.num_simple_radial_calibs);
    }
    if (const ModelData* md = get_md(CameraModelId::kPinhole)) {
      sz.num_pinhole_calibs = get_n(CameraModelId::kPinhole);
      adapters_.at(CameraModelId::kPinhole)
          ->FillSizing(sz, *md, sz.num_pinhole_calibs);
    }
    return sz;
  }

  size_t TotalPoses() const {
    size_t n = 0;
    for (const auto& [_, count] : num_poses_per_model_) {
      n += count;
    }
    return n;
  }

  bool ValidateData() const {
    if (num_points_ == 0 && TotalPoses() == 0) {
      LOG(WARNING) << "No data to optimize";
      return false;
    }
    if (ComputeTotalResiduals() == 0) {
      LOG(WARNING) << "No residuals to optimize";
      return false;
    }
    return true;
  }

#ifndef CASPAR_USE_DOUBLE
  void SetupRigSchurTopology(caspar::GraphSolver& solver) const {
    const ModelData& model_data =
        model_data_per_model_.at(CameraModelId::kPinhole);
    constexpr FactorVariant main_variant =
        FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT;
    constexpr FactorVariant fixed_pose_variant =
        FactorVariant::FIXED_POSE_FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT;
    constexpr FactorVariant fixed_point_variant =
        FactorVariant::FIXED_FOCAL_AND_EXTRA_FIXED_PRINCIPAL_POINT_FIXED_POINT;

    for (int variant_index = 0; variant_index < CASPAR_NUM_VARIANTS;
         ++variant_index) {
      const FactorVariant variant = static_cast<FactorVariant>(variant_index);
      if (model_data.variants[variant_index].num_factors > 0 &&
          variant != main_variant && variant != fixed_pose_variant &&
          variant != fixed_point_variant) {
        LOG(FATAL_THROW)
            << "Caspar rig-Schur received unsupported factor variant "
            << FactorVariantName(variant);
      }
    }

    const VariantData& main =
        model_data.variants[static_cast<int>(main_variant)];
    const VariantData& fixed_pose =
        model_data.variants[static_cast<int>(fixed_pose_variant)];
    const VariantData& fixed_point =
        model_data.variants[static_cast<int>(fixed_point_variant)];
    THROW_CHECK_EQ(main.pose_indices.size(), main.point_indices.size());
    solver.SetRigSchurTopology(main.pose_indices,
                               main.point_indices,
                               fixed_pose.point_indices,
                               fixed_point.pose_indices);
  }
#endif

  std::shared_ptr<BundleAdjustmentSummary> Solve() override {
    if (!ValidateData()) {
      auto summary = std::make_shared<BundleAdjustmentSummary>();
      summary->termination_type = BundleAdjustmentTerminationType::USER_FAILURE;
      return summary;
    }

    caspar::SolverParams<double> params;
    int gpu_index = -1;
    if (options_.caspar) {
      const auto& co = *options_.caspar;
      const std::vector<int> gpu_indices = CSVToVector<int>(co.gpu_index);
      THROW_CHECK_GT(gpu_indices.size(), 0);
      gpu_index = gpu_indices[0];
      params = CreateCasparSolverParameters(co);
    }

    const size_t device_id =
        static_cast<size_t>(gpu_index >= 0 ? gpu_index : FindBestCudaDevice());
    LogFactorDistribution();

    auto solver = CreateSolver(params, BuildSizing(), device_id);
#ifndef CASPAR_USE_DOUBLE
    if (solver_mode_ == CasparSolverMode::RIG_SCHUR) {
      SetupRigSchurTopology(solver);
    }
#endif
    SetupSolverData(solver);
    const bool collect_iters =
        options_.caspar && options_.caspar->collect_iteration_data;
    caspar::SolveResult result;
    if (solver_mode_ == CasparSolverMode::RIG_SCHUR) {
#ifdef CASPAR_USE_DOUBLE
      LOG(FATAL_THROW)
          << "Caspar rig-Schur only supports the default float precision";
#else
      result = solver.solve_rig_schur(
          /*print_progress=*/VLOG_IS_ON(2),
          /*verbose_logging=*/collect_iters);
#endif
    } else {
      result = solver.solve(
          /*print_progress=*/VLOG_IS_ON(2),
          /*verbose_logging=*/collect_iters);
    }
    ReadSolverResults(solver);
    WriteResultsToReconstruction();

    auto summary = CasparBundleAdjustmentSummary::Create(result);
    summary->num_residuals = ComputeTotalResiduals();
    return summary;
  }

  NodeHashMap<CameraModelId, std::unique_ptr<ICasparModelAdapter>> adapters_;
  NodeHashMap<CameraModelId, ModelData> model_data_per_model_;
  NodeHashMap<CameraModelId, size_t> calib_num_per_model_;

  // camera_id -> calib index within its model's array
  NodeHashMap<camera_t, size_t> camera_to_calib_index_;
  // (model_id, calib_idx) -> camera_id  (for write-back)
  std::map<std::pair<CameraModelId, size_t>, camera_t> calib_index_to_camera_;

  CasparSolverMode solver_mode_;
  Reconstruction& reconstruction_;
  size_t num_points_ = 0;
  std::vector<StorageType> point3D_data_;

  // Pose pools are per-model so that SimpleRadial and Pinhole factors are
  // never batched into the same Caspar block (reducing shared memory use).
  NodeHashMap<CameraModelId, size_t> num_poses_per_model_;
  NodeHashMap<CameraModelId, std::vector<StorageType>> pose_data_per_model_;
  NodeHashMap<CameraModelId, NodeHashMap<frame_t, size_t>>
      frame_to_pose_index_per_model_;
  NodeHashMap<CameraModelId, NodeHashMap<size_t, frame_t>>
      pose_index_to_frame_per_model_;

  FlatHashMap<point3D_t, size_t> point3D_id_to_idx_;
  std::vector<point3D_t> point3D_idx_to_id_;

  FlatHashSet<frame_t> frames_from_outside_config_;
  FlatHashSet<camera_t> cameras_from_outside_config_;
  FlatHashSet<frame_t> gauge_fixed_frames_;
  FlatHashSet<point3D_t> gauge_fixed_points_;
  FlatHashMap<point3D_t, size_t> point3D_num_observations_;

  struct RigSchurImageCounts {
    size_t variable_points = 0;
    size_t fixed_points = 0;
  };
  FlatHashMap<image_t, RigSchurImageCounts> rig_schur_image_counts_;
};

BundleAdjustmentOptions CreateFixedRigCasparOptions(
    const FixedRigPosePriorBundleAdjustmentOptions& options) {
  BundleAdjustmentOptions bundle_options;
  bundle_options.ceres = options.ceres;
  bundle_options.caspar = options.caspar;
  bundle_options.refine_focal_length = false;
  bundle_options.refine_principal_point = false;
  bundle_options.refine_extra_params = false;
  bundle_options.refine_sensor_from_rig = false;
  bundle_options.refine_rig_from_world = true;
  bundle_options.refine_points3D = true;
  bundle_options.min_track_length = options.min_track_length;
  bundle_options.constant_rig_from_world_rotation = false;
  bundle_options.print_summary = options.print_summary;
  bundle_options.backend = BundleAdjustmentBackend::CASPAR_RIG_SCHUR;
  return bundle_options;
}

#ifndef CASPAR_USE_DOUBLE
class FixedRigPosePriorCasparBundleAdjuster : public BundleAdjuster {
 public:
  FixedRigPosePriorCasparBundleAdjuster(
      const FixedRigPosePriorBundleAdjustmentOptions& options,
      const BundleAdjustmentConfig& config,
      std::vector<PosePrior> pose_priors,
      Reconstruction& reconstruction)
      : BundleAdjuster(CreateFixedRigCasparOptions(options), config),
        fixed_options_(options),
        pose_priors_(std::move(pose_priors)),
        reconstruction_(reconstruction) {
    VLOG(2) << "Creating Caspar fixed-rig pose-prior bundle adjuster";
    THROW_CHECK_EQ(config_.FixedGauge(), BundleAdjustmentGauge::UNSPECIFIED);
    FilterPosePriors();
    normalized_from_metric_ = reconstruction_.Normalize(/*fixed_scale=*/false);
    BuildProblemData();
  }

  std::shared_ptr<BundleAdjustmentSummary> Solve() override {
    CasparSolverSizing sizing;
    sizing.num_pinhole_poses = pose_index_to_frame_.size();
    sizing.num_points = point3D_index_to_id_.size();
    sizing.num_sensor_from_rig_log_scales = 1;
    sizing.num_fixed_rig_pinhole = pose_indices_.size();
    sizing.num_fixed_rig_position_prior = prior_pose_indices_.size();

    caspar::SolverParams<double> params;
    int gpu_index = -1;
    if (fixed_options_.caspar) {
      const CasparBundleAdjustmentOptions& caspar_options =
          *fixed_options_.caspar;
      const std::vector<int> gpu_indices =
          CSVToVector<int>(caspar_options.gpu_index);
      THROW_CHECK_GT(gpu_indices.size(), 0);
      gpu_index = gpu_indices.front();
      params = CreateCasparSolverParameters(caspar_options);
    }

    const size_t device_id =
        static_cast<size_t>(gpu_index >= 0 ? gpu_index : FindBestCudaDevice());
    auto solver = CreateSolver(params, sizing, device_id);
    SetupSolver(solver);
    const bool collect_iterations =
        fixed_options_.caspar && fixed_options_.caspar->collect_iteration_data;
    const caspar::SolveResult result = solver.solve_rig_schur(
        /*print_progress=*/VLOG_IS_ON(2),
        /*verbose_logging=*/collect_iterations);
    ReadSolver(solver);
    WriteResultsToReconstruction();
    reconstruction_.Transform(Inverse(normalized_from_metric_));

    const auto backend_summary = CasparBundleAdjustmentSummary::Create(result);
    auto summary = std::make_shared<FixedRigPosePriorBundleAdjustmentSummary>();
    summary->termination_type = backend_summary->termination_type;
    summary->num_residuals = static_cast<int>(2 * pose_indices_.size() +
                                              3 * prior_pose_indices_.size());
    if (summary->IsSolutionUsable()) {
      summary->sensor_from_rig_scale = std::exp(log_scale_);
      for (const auto& rig_entry : reconstruction_.Rigs()) {
        Rig& rig = reconstruction_.Rig(rig_entry.first);
        for (auto& sensor_entry : rig.NonRefSensors()) {
          std::optional<Rigid3d>& sensor_from_rig = sensor_entry.second;
          THROW_CHECK(sensor_from_rig.has_value());
          sensor_from_rig->translation() *= summary->sensor_from_rig_scale;
        }
      }
    }
    if (fixed_options_.print_summary || VLOG_IS_ON(1)) {
      LOG(INFO) << summary->BriefReport();
    }
    return summary;
  }

 private:
  static void AppendPose(std::vector<StorageType>& output,
                         const Rigid3d& pose) {
    output.push_back(pose.rotation().x());
    output.push_back(pose.rotation().y());
    output.push_back(pose.rotation().z());
    output.push_back(pose.rotation().w());
    output.push_back(pose.translation().x());
    output.push_back(pose.translation().y());
    output.push_back(pose.translation().z());
  }

  void FilterPosePriors() {
    pose_priors_.erase(
        std::remove_if(pose_priors_.begin(),
                       pose_priors_.end(),
                       [this](const PosePrior& pose_prior) {
                         if (!pose_prior.HasPosition() ||
                             pose_prior.corr_data_id.sensor_id.type !=
                                 SensorType::CAMERA ||
                             !config_.HasImage(pose_prior.corr_data_id.id)) {
                           return true;
                         }
                         return !reconstruction_
                                     .Image(pose_prior.corr_data_id.id)
                                     .IsRefInFrame();
                       }),
        pose_priors_.end());
    THROW_CHECK_GE(pose_priors_.size(), 3)
        << "Fixed-rig pose-prior BA requires at least three reference-camera "
           "position priors";
    FlatHashSet<image_t> image_ids;
    for (const PosePrior& pose_prior : pose_priors_) {
      THROW_CHECK(image_ids.insert(pose_prior.corr_data_id.id).second)
          << "Duplicate pose prior for image " << pose_prior.corr_data_id.id;
    }
  }

  size_t GetOrCreatePose(const frame_t frame_id) {
    auto [it, inserted] =
        frame_to_pose_index_.try_emplace(frame_id, pose_index_to_frame_.size());
    if (inserted) {
      pose_index_to_frame_.push_back(frame_id);
      AppendPose(pose_data_, reconstruction_.Frame(frame_id).RigFromWorld());
    }
    return it->second;
  }

  size_t GetOrCreatePoint(const point3D_t point3D_id) {
    auto [it, inserted] = point3D_id_to_index_.try_emplace(
        point3D_id, point3D_index_to_id_.size());
    if (inserted) {
      point3D_index_to_id_.push_back(point3D_id);
      const Eigen::Vector3d& xyz = reconstruction_.Point3D(point3D_id).xyz;
      point_data_.push_back(xyz.x());
      point_data_.push_back(xyz.y());
      point_data_.push_back(xyz.z());
    }
    return it->second;
  }

  bool HasSufficientTrackLength(const point3D_t point3D_id) const {
    return fixed_options_.min_track_length == 0 ||
           reconstruction_.Point3D(point3D_id).track.Length() >=
               static_cast<size_t>(fixed_options_.min_track_length);
  }

  void BuildProblemData() {
    std::vector<image_t> image_ids(config_.Images().begin(),
                                   config_.Images().end());
    std::sort(image_ids.begin(), image_ids.end());
    for (const image_t image_id : image_ids) {
      const Image& image = reconstruction_.Image(image_id);
      const Camera& camera = *image.CameraPtr();
      THROW_CHECK_EQ(camera.model_id, CameraModelId::kPinhole)
          << "Caspar fixed-rig pose-prior BA only supports PINHOLE cameras";
      const unsigned int pose_index =
          static_cast<unsigned int>(GetOrCreatePose(image.FrameId()));
      const Rigid3d sensor_from_rig =
          image.IsRefInFrame() ? Rigid3d{}
                               : image.FramePtr()->RigPtr()->SensorFromRig(
                                     image.DataId().sensor_id);
      for (const Point2D& point2D : image.Points2D()) {
        if (!point2D.HasPoint3D() ||
            config_.IsIgnoredPoint(point2D.point3D_id) ||
            !HasSufficientTrackLength(point2D.point3D_id)) {
          continue;
        }
        pose_indices_.push_back(pose_index);
        point_indices_.push_back(
            static_cast<unsigned int>(GetOrCreatePoint(point2D.point3D_id)));
        AppendPose(sensor_from_rig_data_, sensor_from_rig);
        for (const double parameter : camera.params) {
          calib_data_.push_back(parameter);
        }
        pixels_.push_back(point2D.xy.x());
        pixels_.push_back(point2D.xy.y());
      }
    }
    THROW_CHECK_GT(pose_indices_.size(), 0);
    const FlatHashSet<unsigned int> active_pose_indices(pose_indices_.begin(),
                                                        pose_indices_.end());
    const auto anchor_image = std::find_if(
        image_ids.begin(), image_ids.end(), [&](const image_t image_id) {
          const Image& image = reconstruction_.Image(image_id);
          return image.IsRefInFrame() &&
                 active_pose_indices.count(static_cast<unsigned int>(
                     frame_to_pose_index_.at(image.FrameId()))) > 0;
        });
    THROW_CHECK(anchor_image != image_ids.end());
    const frame_t rotation_anchor_frame_id =
        reconstruction_.Image(*anchor_image).FrameId();
    rotation_anchor_pose_index_ = static_cast<unsigned int>(
        frame_to_pose_index_.at(rotation_anchor_frame_id));
    BuildPriorData(active_pose_indices);
  }

  void BuildPriorData(const FlatHashSet<unsigned int>& active_pose_indices) {
    const Eigen::Matrix3d normalized_from_metric_scaled_rotation =
        normalized_from_metric_.scale() *
        normalized_from_metric_.rotation().toRotationMatrix();
    for (const PosePrior& pose_prior : pose_priors_) {
      const Image& image = reconstruction_.Image(pose_prior.corr_data_id.id);
      const unsigned int pose_index =
          static_cast<unsigned int>(frame_to_pose_index_.at(image.FrameId()));
      if (active_pose_indices.count(pose_index) == 0) {
        continue;
      }
      prior_pose_indices_.push_back(pose_index);
      const Eigen::Vector3d position =
          normalized_from_metric_ * pose_prior.position;
      prior_positions_.insert(
          prior_positions_.end(), position.data(), position.data() + 3);
      const double fallback_stddev =
          fixed_options_.prior_position_fallback_stddev;
      const Eigen::Matrix3d position_covariance =
          pose_prior.HasPositionCov()
              ? pose_prior.position_covariance
              : fallback_stddev * fallback_stddev * Eigen::Matrix3d::Identity();
      const Eigen::Matrix3d normalized_covariance =
          normalized_from_metric_scaled_rotation * position_covariance *
          normalized_from_metric_scaled_rotation.transpose();
      const Eigen::Matrix3d sqrt_information =
          normalized_covariance.inverse().llt().matrixL().transpose();
      prior_sqrt_information_.insert(prior_sqrt_information_.end(),
                                     sqrt_information.data(),
                                     sqrt_information.data() + 9);
    }
    THROW_CHECK_GE(prior_pose_indices_.size(), 3)
        << "Fixed-rig pose-prior BA requires at least three parameterized "
           "reference-camera position priors";
  }

  void SetupSolver(caspar::GraphSolver& solver) {
    solver.SetPinholePoseNodesFromStackedHost(
        pose_data_.data(), 0, pose_index_to_frame_.size());
    solver.SetPointNodesFromStackedHost(
        point_data_.data(), 0, point3D_index_to_id_.size());
    solver.SetSensorFromRigLogScaleNodesFromStackedHost(&log_scale_, 0, 1);
    solver.SetFixedRigPinholeNum(pose_indices_.size());
    solver.SetFixedRigPinholePoseIndicesFromHost(pose_indices_.data(),
                                                 pose_indices_.size());
    solver.SetFixedRigPinholePointIndicesFromHost(point_indices_.data(),
                                                  point_indices_.size());
    solver.SetFixedRigPinholeSensorFromRigDataFromStackedHost(
        sensor_from_rig_data_.data(), 0, pose_indices_.size());
    solver.SetFixedRigPinholeCalibDataFromStackedHost(
        calib_data_.data(), 0, pose_indices_.size());
    solver.SetFixedRigPinholePixelDataFromStackedHost(
        pixels_.data(), 0, pose_indices_.size());
    const StorageType reprojection_loss_scale =
        fixed_options_.caspar->fixed_rig_reprojection_loss_scale;
    solver.SetFixedRigPinholeReprojectionLossScaleDataFromStackedHost(
        &reprojection_loss_scale);
    solver.SetFixedRigPositionPriorNum(prior_pose_indices_.size());
    solver.SetFixedRigPositionPriorPoseIndicesFromHost(
        prior_pose_indices_.data(), prior_pose_indices_.size());
    solver.SetFixedRigPositionPriorPositionDataFromStackedHost(
        prior_positions_.data(), 0, prior_pose_indices_.size());
    solver.SetFixedRigPositionPriorSqrtInformationDataFromStackedHost(
        prior_sqrt_information_.data(), 0, prior_pose_indices_.size());
    const StorageType prior_position_loss_scale =
        fixed_options_.caspar->prior_position_loss_scale;
    solver.SetFixedRigPositionPriorPositionLossScaleDataFromStackedHost(
        &prior_position_loss_scale);
    solver.SetFixedRigSchurTopology(
        pose_indices_, point_indices_, {}, rotation_anchor_pose_index_);
    solver.finish_indices();
  }

  void ReadSolver(caspar::GraphSolver& solver) {
    solver.GetPinholePoseNodesToStackedHost(
        pose_data_.data(), 0, pose_index_to_frame_.size());
    solver.GetPointNodesToStackedHost(
        point_data_.data(), 0, point3D_index_to_id_.size());
    solver.GetSensorFromRigLogScaleNodesToStackedHost(&log_scale_, 0, 1);
  }

  void WriteResultsToReconstruction() {
    for (size_t index = 0; index < pose_index_to_frame_.size(); ++index) {
      Rigid3d& pose =
          reconstruction_.Frame(pose_index_to_frame_[index]).RigFromWorld();
      pose.rotation().x() = pose_data_[7 * index];
      pose.rotation().y() = pose_data_[7 * index + 1];
      pose.rotation().z() = pose_data_[7 * index + 2];
      pose.rotation().w() = pose_data_[7 * index + 3];
      pose.translation().x() = pose_data_[7 * index + 4];
      pose.translation().y() = pose_data_[7 * index + 5];
      pose.translation().z() = pose_data_[7 * index + 6];
      pose.rotation().normalize();
    }
    for (size_t index = 0; index < point3D_index_to_id_.size(); ++index) {
      Eigen::Vector3d& xyz =
          reconstruction_.Point3D(point3D_index_to_id_[index]).xyz;
      xyz.x() = point_data_[3 * index];
      xyz.y() = point_data_[3 * index + 1];
      xyz.z() = point_data_[3 * index + 2];
    }
  }

  FixedRigPosePriorBundleAdjustmentOptions fixed_options_;
  std::vector<PosePrior> pose_priors_;
  Reconstruction& reconstruction_;
  Sim3d normalized_from_metric_;
  StorageType log_scale_ = 0;
  FlatHashMap<frame_t, size_t> frame_to_pose_index_;
  std::vector<frame_t> pose_index_to_frame_;
  std::vector<StorageType> pose_data_;
  FlatHashMap<point3D_t, size_t> point3D_id_to_index_;
  std::vector<point3D_t> point3D_index_to_id_;
  std::vector<StorageType> point_data_;
  std::vector<unsigned int> pose_indices_;
  std::vector<unsigned int> point_indices_;
  std::vector<StorageType> sensor_from_rig_data_;
  std::vector<StorageType> calib_data_;
  std::vector<StorageType> pixels_;
  std::vector<unsigned int> prior_pose_indices_;
  std::vector<StorageType> prior_positions_;
  std::vector<StorageType> prior_sqrt_information_;
  unsigned int rotation_anchor_pose_index_ = 0;
};
#endif
}  // namespace

caspar::SolverParams<double> CreateCasparSolverParameters(
    const CasparSolverOptions& options) {
  caspar::SolverParams<double> parameters;
  parameters.solver_iter_max = options.solver_iter_max;
  parameters.pcg_iter_max = options.pcg_iter_max;
  parameters.diag_init = options.diag_init;
  parameters.diag_min = options.diag_min;
  parameters.diag_scaling_up = options.diag_scaling_up;
  parameters.diag_scaling_down = options.diag_scaling_down;
  parameters.diag_exit_value = options.diag_exit_value;
  parameters.score_exit_value = options.score_exit_value;
  parameters.pcg_rel_error_exit = options.pcg_rel_error_exit;
  parameters.pcg_rel_score_exit = options.pcg_rel_score_exit;
  parameters.pcg_rel_decrease_min = options.pcg_rel_decrease_min;
  parameters.solver_rel_decrease_min = options.solver_rel_decrease_min;
  return parameters;
}

std::shared_ptr<CasparBundleAdjustmentSummary>
CasparBundleAdjustmentSummary::Create(
    const caspar::SolveResult& caspar_summary) {
  auto summary = std::make_shared<CasparBundleAdjustmentSummary>();
  summary->iteration_count = caspar_summary.iteration_count;
  summary->initial_score = caspar_summary.initial_score;
  summary->final_score = caspar_summary.final_score;
  summary->runtime = caspar_summary.runtime;
  summary->iterations = caspar_summary.iterations;
  switch (caspar_summary.exit_reason) {
    case caspar::ExitReason::CONVERGED_DIAG_EXIT:
      VLOG(1) << "Caspar: CONVERGED_DIAG_EXIT after "
              << caspar_summary.iteration_count << " iters"
              << " (diag limit hit -> likely premature termination)";
      summary->termination_type = BundleAdjustmentTerminationType::CONVERGENCE;
      break;
    case caspar::ExitReason::CONVERGED_SCORE_THRESHOLD:
      VLOG(1) << "Caspar: CONVERGED_SCORE_THRESHOLD after "
              << caspar_summary.iteration_count << " iters";
      summary->termination_type = BundleAdjustmentTerminationType::CONVERGENCE;
      break;
    case caspar::ExitReason::MAX_ITERATIONS:
      VLOG(1) << "Caspar: MAX_ITERATIONS (" << caspar_summary.iteration_count
              << ")";
      summary->termination_type =
          BundleAdjustmentTerminationType::NO_CONVERGENCE;
      break;
    default:
      summary->termination_type = BundleAdjustmentTerminationType::FAILURE;
  }
  return summary;
}

std::unique_ptr<BundleAdjuster> CreateDefaultCasparBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    Reconstruction& reconstruction) {
  return std::make_unique<CasparBundleAdjuster>(
      options, config, reconstruction, CasparSolverMode::STANDARD);
}

std::unique_ptr<BundleAdjuster> CreateDefaultCasparRigSchurBundleAdjuster(
    const BundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    Reconstruction& reconstruction) {
  return std::make_unique<CasparBundleAdjuster>(
      options, config, reconstruction, CasparSolverMode::RIG_SCHUR);
}

std::unique_ptr<BundleAdjuster>
CreateFixedRigPosePriorCasparRigSchurBundleAdjuster(
    const FixedRigPosePriorBundleAdjustmentOptions& options,
    const BundleAdjustmentConfig& config,
    std::vector<PosePrior> pose_priors,
    Reconstruction& reconstruction) {
#ifdef CASPAR_USE_DOUBLE
  LOG(FATAL_THROW) << "Caspar fixed-rig pose-prior BA requires float precision";
  return nullptr;
#else
  return std::make_unique<FixedRigPosePriorCasparBundleAdjuster>(
      options, config, std::move(pose_priors), reconstruction);
#endif
}

}  // namespace colmap
