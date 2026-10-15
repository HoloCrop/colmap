# Usage:
#   ./regenerate-caspar <out_dir> f32|f64 [python_executable]

import inspect
import sys
from itertools import combinations
from pathlib import Path
from typing import Annotated, get_type_hints

# Must be set before importing symforce.symbolic
precision = sys.argv[2] if len(sys.argv) > 2 else "f32"

if precision not in ("f32", "f64"):
    print(f"ERROR: Unknown precision '{precision}'. Expected f32 or f64.")
    sys.exit(1)

import symforce  # noqa: E402

symforce.set_epsilon_to_number(1e-15 if precision == "f64" else 1e-6)

import symforce.symbolic as sf  # noqa: E402
from symforce import typing as T  # noqa: E402
from symforce.caspar import CasparLibrary, SharedConstantPool  # noqa: E402
from symforce.caspar import memory as mem  # noqa: E402
from symforce.caspar.code_formulation import ftypes  # noqa: E402


class CasparEuler(ftypes.Func):
    """Euler's number, missing from Caspar's expression lowering."""

    def assign_code(self, outs, args, dtype):
        return f"{outs[0]} = 2.71828182845904523536;"


ftypes.EXPR_TO_FUNC[ftypes.symengine_wrapper.Exp1] = CasparEuler


# Point and ConstPoint are shared across camera models so that different
# camera types can observe the same 3D points in the same node pool.
class Point(sf.V3):
    pass


class ConstPoint(sf.V3):
    pass


class ConstPixel(sf.V2):
    pass


class ConstReprojectionLossScale(sf.V1):
    pass


class ConstImageFromWorld(sf.M34):
    pass


class ConstPinholeSensorCalibration(sf.Matrix):
    """Fixed sensor-from-rig Pose3 storage followed by PINHOLE calibration."""

    SHAPE = (11, 1)


class SensorFromRigLogScale(sf.V1):
    pass


class ConstReferencePosition(sf.V3):
    pass


class ConstPositionSqrtInformation(sf.V9):
    pass


class ConstPositionLossScale(sf.V1):
    pass


class ConstLogScalePriorTarget(sf.V1):
    pass


class ConstLogScalePriorSqrtInformation(sf.V1):
    pass


class ConstRigPoint(sf.V3):
    pass  # a point fixed in its rig frame


class ConstSurfacePosition(sf.V3):
    pass


class ConstSurfaceSqrtWeight(sf.V1):
    pass


class ConstSurfaceRobust(sf.V2):
    pass  # [Geman-McClure scale, margin]


# Calibration node layout:
#
# When both focal_and_extra/focal and principal_point are tunable, they
# are merged into a single V4 Calib node to save one shared-memory slot per
# block.
# This covers 4 variants: BASE, FIXED_POSE, FIXED_POINT, FIXED_POSE_FIXED_POINT.
#
# When at least one group is fixed, split V2 nodes are used (11 variants).
# Total: 4 merged + 11 split = 15 variants per camera model.
#
# Pose and Calib nodes are camera-model-specific to prevent cross-model
# batching.
# Check that generated kernels stay within the 48 KB shared memory limit if
# adding new parameter groups.


# SimpleRadial: params = [f, cx, cy, k]
class SimpleRadialPose(sf.Pose3):
    pass


class ConstSimpleRadialPose(sf.Pose3):
    pass


class SimpleRadialCalib(sf.V4):
    pass  # [f, k, cx, cy]  (merged)


class ConstSimpleRadialCalib(sf.V4):
    pass


class SimpleRadialPrincipalPoint(sf.V2):
    pass  # [cx, cy]  (split: pp tunable)


class ConstSimpleRadialPrincipalPoint(sf.V2):
    pass


class SimpleRadialFocalAndExtra(sf.V2):
    pass  # [f, k]    (split: focal tunable)


class ConstSimpleRadialFocalAndExtra(sf.V2):
    pass


# Pinhole: params = [fx, fy, cx, cy]
class PinholePose(sf.Pose3):
    pass


class ConstPinholePose(sf.Pose3):
    pass


class PinholeCalib(sf.V4):
    pass  # [fx, fy, cx, cy]  (merged)


class ConstPinholeCalib(sf.V4):
    pass


class PinholePrincipalPoint(sf.V2):
    pass  # [cx, cy]  (split: pp tunable)


class ConstPinholePrincipalPoint(sf.V2):
    pass


class PinholeFocal(sf.V2):
    pass  # [fx, fy]  (split: focal tunable)


class ConstPinholeFocal(sf.V2):
    pass


# Constant sensor-from-rig calibration, stored as ConstantSequential
# (7 floats = 28 B f32 / 56 B f64 loaded from global memory per factor).
# ConstantShared would deduplicate to one slot per unique sensor per block,
# eliminating the per-factor global loads, but Caspar incorrectly counts
# ConstantShared nodes against the 48 KB shared-memory budget even though they
# are never loaded into shared memory.  Switch to ConstantShared once fixed.
# This choice does not affect Jacobian correctness: rig linkage is encoded
# through the shared rig_from_world pose node (same pose index for all sensors
# in a frame), not through sensor_from_rig.
class ConstSimpleRadialSensorFromRig(sf.Pose3):
    pass


class ConstPinholeSensorFromRig(sf.Pose3):
    pass


def _make_variant(
    core_fn, name: str, base_params: list, hints: dict, fixed: dict
):
    new_hints = {}
    for p in base_params:
        if p in fixed:
            new_hints[p] = Annotated[fixed[p], mem.ConstantSequential]
        else:
            new_hints[p] = hints[p]

    tunable_params = [p for p in base_params if p not in fixed]
    const_params = [p for p in base_params if p in fixed]
    ordered = tunable_params + const_params

    # Caspar calls factors as fn(**symbolic_args), so the wrapper accepts both
    # positional and keyword arguments.
    def wrapper(*args, **kwargs):
        merged = {**dict(zip(ordered, args, strict=False)), **kwargs}
        return core_fn(*[merged[p] for p in base_params])

    wrapper.__name__ = name
    wrapper.__annotations__ = {p: new_hints[p] for p in ordered}
    wrapper.__annotations__["return"] = hints.get("return")
    wrapper.__signature__ = inspect.Signature(
        [
            inspect.Parameter(p, inspect.Parameter.POSITIONAL_OR_KEYWORD)
            for p in ordered
        ]
    )
    return wrapper


def register_camera_model(
    caslib,
    model_name: str,
    core_fn,
    fixable_params: dict,
    must_fix_one_of: T.Optional[set] = None,
    include_all_fixed: bool = False,
):
    hints = get_type_hints(core_fn, include_extras=True)
    base_params = list(inspect.signature(core_fn).parameters.keys())
    fixable_items = list(fixable_params.items())

    # include_all_fixed extends the range to N for merged-calib models, where
    # the "all-fixed" subset still has a tunable Calib node.
    max_r = len(fixable_items) + (1 if include_all_fixed else 0)
    for r in range(max_r):
        for combo in combinations(fixable_items, r):
            fixed = dict(combo)

            # Skip split variants where both calib groups are tunable, as those
            # are handled by the merged-calib registration.
            if must_fix_one_of and not any(p in fixed for p in must_fix_one_of):
                continue

            if fixed:
                # Suffix order follows fixable_params definition order to
                # match the C++ dispatch naming.
                suffix = "_".join(
                    f"fixed_{p}" for p, _ in fixable_items if p in fixed
                )
                name = f"{model_name}_{suffix}"
            else:
                name = model_name

            caslib.add_factor(
                _make_variant(core_fn, name, base_params, hints, fixed)
            )


# --- Camera models ---

# Merged cores define the canonical projection math reused by split variants.


def simple_radial_core(
    pose: T.Annotated[SimpleRadialPose, mem.TunableShared],
    sensor_from_rig: T.Annotated[
        ConstSimpleRadialSensorFromRig, mem.ConstantSequential
    ],
    calib: T.Annotated[SimpleRadialCalib, mem.TunableShared],  # [f, k, cx, cy]
    point: T.Annotated[Point, mem.TunableShared],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
) -> sf.V2:
    """Reprojection residual for COLMAP's SIMPLE_RADIAL model (sensor/models.h).

    calib = [f, k, cx, cy]: single focal length, one radial distortion
    coefficient, and principal point.
    pose holds rig_from_world; sensor_from_rig is identity for single-camera
    rigs (cam_from_world == rig_from_world in that case).
    """
    cam_T_world = sensor_from_rig * pose
    f, k, cx, cy = calib
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    r = 1 + k * p.squared_norm()
    return f * r * p + sf.V2([cx, cy]) - pixel


def pinhole_core(
    pose: T.Annotated[PinholePose, mem.TunableShared],
    sensor_from_rig: T.Annotated[
        ConstPinholeSensorFromRig, mem.ConstantSequential
    ],
    calib: T.Annotated[PinholeCalib, mem.TunableShared],  # [fx, fy, cx, cy]
    point: T.Annotated[Point, mem.TunableShared],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
) -> sf.V2:
    """Reprojection residual for COLMAP's PINHOLE model (sensor/models.h).

    calib = [fx, fy, cx, cy]: two independent focal lengths and principal point,
    no distortion.
    pose holds rig_from_world; sensor_from_rig is identity for single-camera
    rigs (cam_from_world == rig_from_world in that case).
    """
    cam_T_world = sensor_from_rig * pose
    fx, fy, cx, cy = calib
    point_cam = cam_T_world * point
    depth = point_cam[2]
    p = sf.V2(point_cam[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
    return sf.V2([fx * p[0] + cx, fy * p[1] + cy]) - pixel


def pseudo_huber_norm(residual: sf.Matrix, loss_scale: sf.V1) -> sf.Matrix:
    """Apply one rotationally invariant pseudo-Huber loss to a residual."""
    scaled_squared_norm = residual.squared_norm() / loss_scale[0] ** 2
    weight = sf.sqrt(2 / (sf.sqrt(1 + scaled_squared_norm) + 1))
    return weight * residual


def fixed_rig_pinhole(
    pose: T.Annotated[PinholePose, mem.TunableShared],
    sensor_from_rig: T.Annotated[
        ConstPinholeSensorFromRig, mem.ConstantSequential
    ],
    sensor_from_rig_log_scale: T.Annotated[
        SensorFromRigLogScale, mem.TunableUnique
    ],
    calib: T.Annotated[ConstPinholeCalib, mem.ConstantSequential],
    point: T.Annotated[Point, mem.TunableShared],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
    reprojection_loss_scale: T.Annotated[
        ConstReprojectionLossScale, mem.ConstantUnique
    ],
) -> sf.V2:
    """Robust fixed-calibration reprojection with one shared baseline scale."""
    scaled_sensor_from_rig = sf.Pose3(
        sensor_from_rig.R,
        sf.exp(sensor_from_rig_log_scale[0]) * sensor_from_rig.t,
    )
    residual = pinhole_core(pose, scaled_sensor_from_rig, calib, point, pixel)
    return pseudo_huber_norm(residual, reprojection_loss_scale)


def row_fixed_rig_pinhole(
    pose: T.Annotated[PinholePose, mem.TunableShared],
    sensor_calibration: T.Annotated[
        ConstPinholeSensorCalibration, mem.ConstantShared
    ],
    sensor_from_rig_log_scale: T.Annotated[
        SensorFromRigLogScale, mem.TunableUnique
    ],
    point: T.Annotated[Point, mem.TunableShared],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
    reprojection_loss_scale: T.Annotated[
        ConstReprojectionLossScale, mem.ConstantUnique
    ],
) -> sf.V2:
    """Robust row reprojection with fixed indexed calibration and live scale."""
    sensor_from_rig = sf.Pose3.from_storage(sensor_calibration[:7])
    calib = sf.V4(sensor_calibration[7:])
    scaled_sensor_from_rig = sf.Pose3(
        sensor_from_rig.R,
        sf.exp(sensor_from_rig_log_scale[0]) * sensor_from_rig.t,
    )
    residual = pinhole_core(pose, scaled_sensor_from_rig, calib, point, pixel)
    return pseudo_huber_norm(residual, reprojection_loss_scale)


def fixed_rig_position_prior(
    pose: T.Annotated[PinholePose, mem.TunableShared],
    position: T.Annotated[ConstReferencePosition, mem.ConstantSequential],
    sqrt_information: T.Annotated[
        ConstPositionSqrtInformation, mem.ConstantSequential
    ],
    position_loss_scale: T.Annotated[
        ConstPositionLossScale, mem.ConstantUnique
    ],
) -> sf.V3:
    """Robust covariance-weighted reference-camera position prior."""
    weight = sf.Matrix33.from_storage(sqrt_information.to_storage())
    residual = weight * (pose.inverse().t - position)
    return pseudo_huber_norm(residual, position_loss_scale)


def fixed_rig_sensor_position_prior(
    pose: T.Annotated[PinholePose, mem.TunableShared],
    sensor_from_rig: T.Annotated[
        ConstPinholeSensorFromRig, mem.ConstantSequential
    ],
    sensor_from_rig_log_scale: T.Annotated[
        SensorFromRigLogScale, mem.TunableUnique
    ],
    position: T.Annotated[ConstReferencePosition, mem.ConstantSequential],
    sqrt_information: T.Annotated[
        ConstPositionSqrtInformation, mem.ConstantSequential
    ],
) -> sf.V3:
    """Prior on a virtual rig sensor position with live baseline scale."""
    scaled_sensor_from_rig = sf.Pose3(
        sensor_from_rig.R,
        sf.exp(sensor_from_rig_log_scale[0]) * sensor_from_rig.t,
    )
    sensor_from_world = scaled_sensor_from_rig * pose
    weight = sf.Matrix33.from_storage(sqrt_information.to_storage())
    return weight * (sensor_from_world.inverse().t - position)


def fixed_rig_log_scale_prior(
    sensor_from_rig_log_scale: T.Annotated[
        SensorFromRigLogScale, mem.TunableUnique
    ],
    target: T.Annotated[ConstLogScalePriorTarget, mem.ConstantUnique],
    sqrt_information: T.Annotated[
        ConstLogScalePriorSqrtInformation, mem.ConstantUnique
    ],
) -> sf.V1:
    """Gaussian prior on the shared live log baseline scale."""
    return sqrt_information * (sensor_from_rig_log_scale - target)


def fixed_camera_pinhole_point(
    point: T.Annotated[Point, mem.TunableShared],
    image_from_world: T.Annotated[ConstImageFromWorld, mem.ConstantShared],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
) -> sf.V2:
    """Squared point-only reprojection with fixed indexed camera matrices."""
    point_homogeneous = sf.V4([point[0], point[1], point[2], 1])
    projected = image_from_world * point_homogeneous
    depth = projected[2]
    return (
        sf.V2(projected[:2]) / (depth + sf.epsilon() * sf.sign_no_zero(depth))
        - pixel
    )


def rig_surface(
    pose: T.Annotated[PinholePose, mem.TunableShared],
    rig_point: T.Annotated[ConstRigPoint, mem.ConstantSequential],
    position: T.Annotated[ConstSurfacePosition, mem.ConstantSequential],
    sqrt_weight: T.Annotated[ConstSurfaceSqrtWeight, mem.ConstantSequential],
    robust: T.Annotated[ConstSurfaceRobust, mem.ConstantUnique],
) -> sf.V3:
    """Robust distance from a point carried by a rig frame to a fixed position.

    pose holds rig_from_world. The offset's length beyond the margin goes
    through a Geman-McClure kernel; sqrt_weight scales the result against the
    other factors' units.
    """
    offset = pose.inverse() * rig_point - position
    distance = sf.sqrt(offset.squared_norm() + sf.epsilon())
    scale, margin = robust
    outside = sf.Max(distance - margin, 0)
    robust_distance = scale * outside / sf.sqrt(outside**2 + scale**2)
    return sqrt_weight[0] * robust_distance / distance * offset


# Split cores delegate to merged cores to avoid duplicating projection math.


def simple_radial_split_core(
    pose: T.Annotated[SimpleRadialPose, mem.TunableShared],
    sensor_from_rig: T.Annotated[
        ConstSimpleRadialSensorFromRig, mem.ConstantSequential
    ],
    focal_and_extra: T.Annotated[SimpleRadialFocalAndExtra, mem.TunableShared],
    principal_point: T.Annotated[SimpleRadialPrincipalPoint, mem.TunableShared],
    point: T.Annotated[Point, mem.TunableShared],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
) -> sf.V2:
    """Split-calib variant of simple_radial_core.

    For COLMAP's SIMPLE_RADIAL model. Used when focal/extra and principal
    point are tuned independently.
    focal_and_extra = [f, k], principal_point = [cx, cy].
    """
    calib = sf.V4(
        [
            focal_and_extra[0],
            focal_and_extra[1],
            principal_point[0],
            principal_point[1],
        ]
    )
    return simple_radial_core(pose, sensor_from_rig, calib, point, pixel)


def pinhole_split_core(
    pose: T.Annotated[PinholePose, mem.TunableShared],
    sensor_from_rig: T.Annotated[
        ConstPinholeSensorFromRig, mem.ConstantSequential
    ],
    focal: T.Annotated[PinholeFocal, mem.TunableShared],
    principal_point: T.Annotated[PinholePrincipalPoint, mem.TunableShared],
    point: T.Annotated[Point, mem.TunableShared],
    pixel: T.Annotated[ConstPixel, mem.ConstantSequential],
) -> sf.V2:
    """Split-calib variant of pinhole_core for COLMAP's PINHOLE model.

    Used when focal lengths and principal point are tuned independently.
    focal = [fx, fy], principal_point = [cx, cy].
    """
    calib = sf.V4([focal[0], focal[1], principal_point[0], principal_point[1]])
    return pinhole_core(pose, sensor_from_rig, calib, point, pixel)


dtype = mem.DType.DOUBLE if precision == "f64" else mem.DType.FLOAT
caslib = CasparLibrary(name="caspar_lib", dtype=dtype)


# Suffix order defines generated variant names and must match the C++ dispatch
# logic that builds names from BundleAdjustmentOptions.
#
# COLMAP flag mapping:
#   refine_rig_from_world                      -> pose
#   refine_focal_length && refine_extra_params -> focal_and_extra / focal
#   refine_principal_point                     -> principal_point
#   refine_points3D                            -> point
#
# Limitations:
#   - constant_rig_from_world_rotation not supported (needs separate pose
#     rotation/translation sub-nodes)
#   - refine_sensor_from_rig=true not supported (tunable sensor_from_rig would
#     need a second Pose3 node, likely overflowing the 48 KB limit without
#     splitting Pose3 into rotation/translation sub-nodes)
#   - refine_focal_length != refine_extra_params not supported (observations
#     skipped with a warning because the merged focal_and_extra node
#     cannot be split)
#
# sensor_from_rig is always present as a ConstantShared parameter. For single-
# camera datasets the host passes identity, making it a no-op. For multi-camera
# rigs it carries the per-sensor extrinsic calibration.

FIXABLE_SIMPLE_RADIAL = {
    "pose": ConstSimpleRadialPose,
    "point": ConstPoint,
}

FIXABLE_PINHOLE = {
    "pose": ConstPinholePose,
    "point": ConstPoint,
}

FIXABLE_SIMPLE_RADIAL_SPLIT = {
    "pose": ConstSimpleRadialPose,
    "focal_and_extra": ConstSimpleRadialFocalAndExtra,
    "principal_point": ConstSimpleRadialPrincipalPoint,
    "point": ConstPoint,
}

FIXABLE_PINHOLE_SPLIT = {
    "pose": ConstPinholePose,
    "focal": ConstPinholeFocal,
    "principal_point": ConstPinholePrincipalPoint,
    "point": ConstPoint,
}

# Merged: BASE, FIXED_POSE, FIXED_POINT, FIXED_POSE_FIXED_POINT (4 variants).
register_camera_model(
    caslib,
    "simple_radial",
    simple_radial_core,
    FIXABLE_SIMPLE_RADIAL,
    include_all_fixed=True,
)
register_camera_model(
    caslib, "pinhole", pinhole_core, FIXABLE_PINHOLE, include_all_fixed=True
)
caslib.add_factor(fixed_rig_pinhole)
caslib.add_factor(fixed_rig_position_prior)
caslib.add_factor(fixed_rig_sensor_position_prior)
caslib.add_factor(fixed_rig_log_scale_prior)
caslib.add_factor(
    row_fixed_rig_pinhole,
    shared_constant_pools={
        "sensor_calibration": SharedConstantPool(ConstPinholeSensorCalibration)
    },
)
caslib.add_factor(
    fixed_camera_pinhole_point,
    shared_constant_pools={
        "image_from_world": SharedConstantPool(ConstImageFromWorld)
    },
)

# Split: all variants where at least one of
# {focal_and_extra, principal_point} is fixed (11 variants per model).
register_camera_model(
    caslib,
    "simple_radial_split",
    simple_radial_split_core,
    FIXABLE_SIMPLE_RADIAL_SPLIT,
    must_fix_one_of={"focal_and_extra", "principal_point"},
)
register_camera_model(
    caslib,
    "pinhole_split",
    pinhole_split_core,
    FIXABLE_PINHOLE_SPLIT,
    must_fix_one_of={"focal", "principal_point"},
)

# Registered last: its count is the final CreateSolver factor argument.
caslib.add_factor(rig_surface)

out_dir = Path(f"{sys.argv[1]}")
print(f"Generating Caspar kernels with precision {precision}: {out_dir}")
caslib.generate(out_dir, use_symlinks=False, python_bindings=False)
