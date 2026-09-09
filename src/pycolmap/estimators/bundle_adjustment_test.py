import numpy as np
import pytest

import pycolmap

caspar_only = pytest.mark.skipif(
    not hasattr(pycolmap, "caspar_refine_pinhole_points"),
    reason="requires a CASPAR-enabled build",
)


def test_bundle_adjustment_termination_type_enum():
    assert {
        k: int(v)
        for k, v in pycolmap.BundleAdjustmentTerminationType.__members__.items()
    } == {
        "CONVERGENCE": 0,
        "NO_CONVERGENCE": 1,
        "FAILURE": 2,
        "USER_SUCCESS": 3,
        "USER_FAILURE": 4,
    }


def test_bundle_adjustment_gauge_enum():
    assert {
        k: int(v) for k, v in pycolmap.BundleAdjustmentGauge.__members__.items()
    } == {
        "UNSPECIFIED": -1,
        "TWO_CAMS_FROM_WORLD": 0,
        "THREE_POINTS": 1,
    }


def test_bundle_adjustment_backend_enum():
    assert {
        k: int(v)
        for k, v in pycolmap.BundleAdjustmentBackend.__members__.items()
    } == {
        "CERES": 0,
        "CASPAR": 1,
    }
    assert (
        pycolmap.BundleAdjustmentBackend("CASPAR")
        == pycolmap.BundleAdjustmentBackend.CASPAR
    )


def test_loss_function_type_enum():
    assert {
        k: int(v) for k, v in pycolmap.LossFunctionType.__members__.items()
    } == {
        "TRIVIAL": 0,
        "SOFT_L1": 1,
        "CAUCHY": 2,
        "HUBER": 3,
    }


def test_bundle_adjustment_summary_default_init():
    summary = pycolmap.BundleAdjustmentSummary()
    assert summary is not None


def test_bundle_adjustment_summary_num_residuals_readwrite():
    summary = pycolmap.BundleAdjustmentSummary()
    summary.num_residuals = 100
    assert summary.num_residuals == 100


def test_bundle_adjustment_summary_termination_type_readwrite():
    summary = pycolmap.BundleAdjustmentSummary()
    summary.termination_type = (
        pycolmap.BundleAdjustmentTerminationType.CONVERGENCE
    )
    assert (
        summary.termination_type
        == pycolmap.BundleAdjustmentTerminationType.CONVERGENCE
    )


def test_bundle_adjustment_summary_is_solution_usable():
    summary = pycolmap.BundleAdjustmentSummary()
    result = summary.is_solution_usable()
    assert isinstance(result, bool)


def test_bundle_adjustment_summary_brief_report():
    summary = pycolmap.BundleAdjustmentSummary()
    report = summary.brief_report()
    assert isinstance(report, str)


def test_ceres_bundle_adjustment_summary_default_init():
    summary = pycolmap.CeresBundleAdjustmentSummary()
    assert summary is not None


def test_ceres_bundle_adjustment_summary_inherits():
    summary = pycolmap.CeresBundleAdjustmentSummary()
    assert isinstance(summary, pycolmap.BundleAdjustmentSummary)


def test_bundle_adjustment_config_default_init():
    config = pycolmap.BundleAdjustmentConfig()
    assert config is not None


def test_bundle_adjustment_config_images_property():
    config = pycolmap.BundleAdjustmentConfig()
    images = config.images
    assert len(images) == 0


def test_ceres_ba_options_default_init():
    options = pycolmap.CeresBundleAdjustmentOptions()
    assert options is not None


def test_ceres_ba_options_check():
    options = pycolmap.CeresBundleAdjustmentOptions()
    result = options.check()
    assert isinstance(result, bool)


def test_caspar_ba_options_default_init():
    options = pycolmap.CasparBundleAdjustmentOptions()
    assert options is not None


def test_caspar_ba_options_readwrite():
    options = pycolmap.CasparBundleAdjustmentOptions()
    options.solver_iter_max = 17
    options.pcg_iter_max = 11
    options.diag_init = 2.0
    options.gpu_index = "0"
    assert options.solver_iter_max == 17
    assert options.pcg_iter_max == 11
    assert options.diag_init == 2.0
    assert options.gpu_index == "0"


def test_ba_options_default_init():
    options = pycolmap.BundleAdjustmentOptions()
    assert options is not None


def test_ba_options_refine_focal_length_readwrite():
    options = pycolmap.BundleAdjustmentOptions()
    original = options.refine_focal_length
    assert isinstance(original, bool)
    options.refine_focal_length = not original
    assert options.refine_focal_length == (not original)


def test_ba_options_refine_principal_point_readwrite():
    options = pycolmap.BundleAdjustmentOptions()
    original = options.refine_principal_point
    assert isinstance(original, bool)
    options.refine_principal_point = not original
    assert options.refine_principal_point == (not original)


def test_ba_options_refine_extra_params_readwrite():
    options = pycolmap.BundleAdjustmentOptions()
    original = options.refine_extra_params
    assert isinstance(original, bool)
    options.refine_extra_params = not original
    assert options.refine_extra_params == (not original)


def test_ba_options_refine_points3d_readwrite():
    options = pycolmap.BundleAdjustmentOptions()
    original = options.refine_points3D
    assert isinstance(original, bool)
    options.refine_points3D = not original
    assert options.refine_points3D == (not original)


def test_ba_options_min_track_length_readwrite():
    options = pycolmap.BundleAdjustmentOptions()
    assert isinstance(options.min_track_length, int)
    options.min_track_length = 5
    assert options.min_track_length == 5


def test_ba_options_print_summary_readwrite():
    options = pycolmap.BundleAdjustmentOptions()
    original = options.print_summary
    assert isinstance(original, bool)
    options.print_summary = not original
    assert options.print_summary == (not original)


def test_ba_options_backend_readwrite():
    options = pycolmap.BundleAdjustmentOptions()
    options.backend = pycolmap.BundleAdjustmentBackend.CERES
    assert options.backend == pycolmap.BundleAdjustmentBackend.CERES
    options.backend = pycolmap.BundleAdjustmentBackend.CASPAR
    assert options.backend == pycolmap.BundleAdjustmentBackend.CASPAR


def test_ba_options_ceres_property():
    options = pycolmap.BundleAdjustmentOptions()
    ceres = options.ceres
    assert isinstance(ceres, pycolmap.CeresBundleAdjustmentOptions)


def test_ba_options_caspar_property():
    options = pycolmap.BundleAdjustmentOptions()
    caspar = options.caspar
    assert isinstance(caspar, pycolmap.CasparBundleAdjustmentOptions)


def test_ba_options_check():
    options = pycolmap.BundleAdjustmentOptions()
    result = options.check()
    assert isinstance(result, bool)


def test_pose_prior_ba_options_default_init():
    options = pycolmap.PosePriorBundleAdjustmentOptions()
    assert options is not None


def test_pose_prior_ba_options_prior_position_fallback_stddev_readwrite():
    options = pycolmap.PosePriorBundleAdjustmentOptions()
    assert isinstance(options.prior_position_fallback_stddev, float)
    options.prior_position_fallback_stddev = 5.0
    assert options.prior_position_fallback_stddev == 5.0


def test_ceres_pose_prior_ba_options_default_init():
    options = pycolmap.CeresPosePriorBundleAdjustmentOptions()
    assert options is not None


def test_ceres_pose_prior_ba_options_check():
    options = pycolmap.CeresPosePriorBundleAdjustmentOptions()
    result = options.check()
    assert isinstance(result, bool)


def test_create_default_bundle_adjuster(synthetic_reconstruction):
    options = pycolmap.BundleAdjustmentOptions()
    config = pycolmap.BundleAdjustmentConfig()
    adjuster = pycolmap.create_default_bundle_adjuster(
        options, config, synthetic_reconstruction
    )
    assert adjuster is not None


def test_create_default_ceres_bundle_adjuster(synthetic_reconstruction):
    options = pycolmap.BundleAdjustmentOptions()
    config = pycolmap.BundleAdjustmentConfig()
    adjuster = pycolmap.create_default_ceres_bundle_adjuster(
        options, config, synthetic_reconstruction
    )
    assert adjuster is not None


def test_bundle_adjustment_pipeline(synthetic_reconstruction):
    reconstruction = pycolmap.Reconstruction(synthetic_reconstruction)
    options = pycolmap.BundleAdjustmentOptions()
    options.print_summary = False
    pycolmap.bundle_adjustment(reconstruction, options)


def pinhole_projections(camera_centers):
    projections = np.zeros((len(camera_centers), 3, 4), dtype=np.float32)
    projections[:, :3, :3] = np.array(
        [[400.0, 0.0, 320.0], [0.0, 400.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    projections[:, 0, 3] = -400.0 * np.asarray(camera_centers, dtype=np.float32)
    return projections


def project_observations(projections, points, point_indices, image_indices):
    homogeneous_points = np.concatenate(
        [points, np.ones((len(points), 1), dtype=np.float32)], axis=1
    )
    projected = np.einsum(
        "nij,nj->ni",
        projections[image_indices],
        homogeneous_points[point_indices],
    )
    return np.ascontiguousarray(projected[:, :2] / projected[:, 2:3])


@caspar_only
def test_caspar_refines_points_against_exact_fixed_cameras():
    projections = pinhole_projections([-1.0, 0.0, 1.0])
    original_projections = projections.copy()
    expected = np.array([[0.2, -0.1, 4.0], [-0.4, 0.3, 5.0]], dtype=np.float32)
    initial = np.array([[0.7, 0.2, 3.2], [-0.8, 0.0, 4.1]], dtype=np.float32)
    point_indices = np.array([0, 1, 0, 1, 0, 1], dtype=np.uint32)
    image_indices = np.array([0, 0, 1, 1, 2, 2], dtype=np.uint32)
    pixels = project_observations(
        projections, expected, point_indices, image_indices
    )

    options = pycolmap.CasparPointRefinementOptions()
    options.solver_iter_max = 100
    result = pycolmap.caspar_refine_pinhole_points(
        initial,
        projections,
        point_indices,
        image_indices,
        pixels,
        options,
    )

    np.testing.assert_allclose(result.points, expected, atol=1e-3)
    np.testing.assert_array_equal(projections, original_projections)
    assert result.summary.final_score < result.summary.initial_score


@caspar_only
@pytest.mark.parametrize("refine_scale", [True, False])
def test_fixed_rig_array_ba_solves_live_sensor_translation_scale(refine_scale):
    rng = np.random.default_rng(7)
    expected_scale = 1.008
    baseline = 0.4
    rig_centers = np.array(
        [[100.0, 250.0, 3.0], [101.1, 250.03, 3.0], [102.2, 249.98, 3.0]],
        dtype=np.float64,
    )
    expected_points = np.column_stack(
        [
            rng.uniform(99.7, 102.5, 24),
            rng.uniform(249.4, 250.6, 24),
            rng.uniform(7.0, 10.0, 24),
        ]
    ).astype(np.float32)
    sensors_from_rig = [
        pycolmap.Rigid3d(),
        pycolmap.Rigid3d(
            np.column_stack([np.eye(3), np.array([-baseline, 0.0, 0.0])])
        ),
    ]
    sensor_calibrations = np.array(
        [[800.0, 805.0, 640.0, 480.0], [800.0, 805.0, 640.0, 480.0]],
        dtype=np.float32,
    )
    cameras = [
        pycolmap.Camera(
            model="PINHOLE",
            width=1280,
            height=960,
            params=calibration,
        )
        for calibration in sensor_calibrations
    ]
    rigs_from_world = [
        pycolmap.Rigid3d(np.column_stack([np.eye(3), -center]))
        for center in rig_centers
    ]
    image_frame_indices = np.repeat(np.arange(3, dtype=np.uint32), 2)
    image_sensor_indices = np.tile(np.arange(2, dtype=np.uint32), 3)
    observation_xy = np.empty(
        (len(image_frame_indices) * len(expected_points), 2), np.float32
    )
    for image_index, (frame_index, sensor_index) in enumerate(
        zip(image_frame_indices, image_sensor_indices, strict=True)
    ):
        start = image_index * len(expected_points)
        end = start + len(expected_points)
        sensor_center = rig_centers[frame_index].copy()
        sensor_center[0] += sensor_index * baseline * expected_scale
        point_camera = expected_points - sensor_center
        observation_xy[start:end] = cameras[sensor_index].img_from_cam(
            point_camera
        )

    initial_points = expected_points + rng.normal(
        0.0, 0.02, expected_points.shape
    ).astype(np.float32)
    prior_frame_indices = np.arange(3, dtype=np.uint32)
    prior_sqrt_information = np.repeat(
        (1000.0 * np.eye(3, dtype=np.float32))[None], 3, axis=0
    )
    options = pycolmap.CasparBundleAdjustmentOptions()
    options.gpu_index = "0"
    options.solver_iter_max = 100
    source_points = np.asfortranarray(initial_points)
    row_source = pycolmap.CasparRowTrackSource(
        source_points,
        np.arange(len(expected_points), dtype=np.int64),
        np.arange(len(expected_points), dtype=np.uint32),
        np.arange(
            0,
            (len(expected_points) + 1) * len(image_frame_indices),
            len(image_frame_indices),
            dtype=np.int64,
        ),
        np.tile(
            np.arange(len(image_frame_indices), dtype=np.int32),
            len(expected_points),
        ),
        np.ascontiguousarray(
            observation_xy.reshape(
                len(image_frame_indices), len(expected_points), 2
            )
            .transpose(1, 0, 2)
            .reshape(-1, 2)
        ),
        np.empty(0, dtype=np.uint32),
        np.arange(len(image_frame_indices), dtype=np.uint32),
    )
    assert np.shares_memory(row_source.points, source_points)
    assert row_source.points.strides == source_points.strides
    if not refine_scale:
        sensors_from_rig[1].translation *= expected_scale
    result = pycolmap.caspar_optimize_row(
        (row_source,),
        np.arange(len(expected_points) + 1, dtype=np.uint32),
        np.arange(len(expected_points), dtype=np.uint32),
        np.full(len(expected_points), len(image_frame_indices), np.uint32),
        np.arange(len(expected_points), dtype=np.uint32),
        rigs_from_world,
        image_frame_indices,
        image_sensor_indices,
        sensors_from_rig,
        sensor_calibrations,
        prior_frame_indices,
        rig_centers.astype(np.float32),
        prior_sqrt_information,
        options,
        refine_scale=refine_scale,
    )

    def reprojection_rmse(sensor_scale):
        squared_error = 0.0
        for image_index, (frame_index, sensor_index) in enumerate(
            zip(image_frame_indices, image_sensor_indices, strict=True)
        ):
            sensor = sensors_from_rig[sensor_index]
            scaled_sensor = pycolmap.Rigid3d(
                sensor.rotation,
                np.asarray(sensor.translation) * sensor_scale,
            )
            sensor_from_world = (
                scaled_sensor * result.rigs_from_world[frame_index]
            )
            predicted = cameras[sensor_index].img_from_cam(
                sensor_from_world * result.points
            )
            start = image_index * len(expected_points)
            end = start + len(expected_points)
            squared_error += np.square(
                predicted - observation_xy[start:end]
            ).sum()
        return np.sqrt(squared_error / len(observation_xy))

    assert result.summary.is_solution_usable()
    assert result.observation_count == len(observation_xy)
    if refine_scale:
        assert result.sensor_from_rig_scale == pytest.approx(
            expected_scale, abs=5e-4
        )
    else:
        assert result.sensor_from_rig_scale == 1.0
    assert result.summary.final_score < result.summary.initial_score
    assert reprojection_rmse(result.sensor_from_rig_scale) < 1e-2
    if refine_scale:
        assert reprojection_rmse(1.0) > 0.1


@caspar_only
def test_row_section_selection_samples_all_sources_and_prefers_long_tracks():
    first = pycolmap.CasparRowTrackSource(
        np.zeros((6, 3), dtype=np.float32),
        np.arange(6, dtype=np.int64),
        np.array((0, 1, 1, 2, 3, 4), dtype=np.uint32),
        np.array((0, 1, 2, 3, 4, 6, 7), dtype=np.int64),
        np.array((0, 1, 1, 2, 2, 0, 0), dtype=np.int32),
        np.array(
            (
                (10, 10),
                (10, 10),
                (11, 11),
                (80, 80),
                (81, 81),
                (81, 81),
                (20, 20),
            ),
            dtype=np.float32,
        ),
        np.empty(0, dtype=np.uint32),
        np.arange(3, dtype=np.uint32),
    )
    second = pycolmap.CasparRowTrackSource(
        np.zeros((1, 3), dtype=np.float32),
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.uint32),
        np.array((0, 1), dtype=np.int64),
        np.zeros(1, dtype=np.int32),
        np.array(((10, 10),), dtype=np.float32),
        np.empty(0, dtype=np.uint32),
        np.array((1,), dtype=np.uint32),
    )
    sources = (first, second)
    point_track_offsets = np.array((0, 2, 4, 5, 6, 7), dtype=np.uint32)
    point_track_indices = np.array((0, 6, 1, 2, 3, 4, 5), dtype=np.uint32)
    source_support = np.array((2, 1, 1, 1, 1), dtype=np.uint16)
    image_frame_indices = np.arange(3, dtype=np.uint32)
    image_sensor_indices = np.zeros(3, dtype=np.uint32)
    sensor_dimensions = np.array(((100, 100),), dtype=np.float32)
    active_frames = np.array((False, True, True))

    selected = pycolmap.caspar_select_row_points(
        sources,
        source_support,
        image_frame_indices,
        image_sensor_indices,
        sensor_dimensions,
        active_frames,
        1,
        1,
    )
    stats = pycolmap.caspar_row_section_stats(
        sources,
        point_track_offsets,
        point_track_indices,
        source_support,
        image_frame_indices,
        image_sensor_indices,
        sensor_dimensions,
        active_frames,
        1,
        np.array((0, 1, 2), dtype=np.uint32),
    )
    plan = pycolmap.caspar_row_sections_plan_stats(
        sources,
        point_track_offsets,
        point_track_indices,
        source_support,
        image_frame_indices,
        image_sensor_indices,
        sensor_dimensions,
        np.array((active_frames, (False, True, False))),
        1,
        10_000,
    )
    zero_density = pycolmap.caspar_select_row_points(
        sources,
        source_support,
        image_frame_indices,
        image_sensor_indices,
        sensor_dimensions,
        active_frames,
        1,
        0,
    )
    denser = pycolmap.caspar_select_row_points(
        sources,
        source_support,
        image_frame_indices,
        image_sensor_indices,
        sensor_dimensions,
        active_frames,
        1,
        2,
    )
    whole_row = pycolmap.caspar_select_row_points(
        sources,
        source_support,
        image_frame_indices,
        image_sensor_indices,
        sensor_dimensions,
        np.ones(3, dtype=np.bool_),
        1,
        1,
    )
    sibling_only = pycolmap.caspar_select_row_points(
        sources,
        source_support,
        image_frame_indices,
        image_sensor_indices,
        sensor_dimensions,
        np.array((False, True, False)),
        1,
        1,
    )

    np.testing.assert_array_equal(selected.point_indices, (0, 1, 3))
    assert selected.quota_truncated
    assert [tier.point_count for tier in stats] == [0, 3, 4]
    assert [tier.observation_count for tier in stats] == [0, 6, 7]
    assert [tier.active_observation_count for tier in stats] == [0, 4, 5]
    assert stats[1].point_count == len(selected.point_indices)
    assert plan.tracks_per_spatial_cell == 2
    assert len(zero_density.point_indices) == 0
    assert zero_density.quota_truncated
    np.testing.assert_array_equal(denser.point_indices, (0, 1, 2, 3))
    assert not denser.quota_truncated
    np.testing.assert_array_equal(whole_row.point_indices, (0, 1, 3))
    assert whole_row.quota_truncated
    np.testing.assert_array_equal(sibling_only.point_indices, (0, 1))
    assert not sibling_only.quota_truncated


@caspar_only
def test_row_section_ba_updates_all_points_observed_by_active_frames():
    rng = np.random.default_rng(11)
    section_centers = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.01, 0.0],
            [2.0, 0.02, 0.0],
            [3.0, 0.01, 0.0],
            [4.0, 0.0, 0.0],
        ]
    )
    all_centers = np.vstack(
        [section_centers, [-1e6, 0.0, 0.0], [1e6, 0.0, 0.0]]
    )
    expected_points = np.column_stack(
        [
            rng.uniform(-0.5, 2.5, 48),
            rng.uniform(-0.8, 0.8, 48),
            rng.uniform(5.0, 8.0, 48),
        ]
    ).astype(np.float32)
    rigs_from_world = [
        pycolmap.Rigid3d(np.column_stack([np.eye(3), -center]))
        for center in all_centers
    ]
    rigs_from_world[2] = pycolmap.Rigid3d(
        np.column_stack([np.eye(3), -np.array([2.18, -0.06, 0.04])])
    )
    sensors_from_rig = [pycolmap.Rigid3d()]
    sensor_calibrations = np.array(
        [[900.0, 905.0, 640.0, 480.0]], dtype=np.float32
    )
    camera = pycolmap.Camera(
        model="PINHOLE",
        width=1280,
        height=960,
        params=sensor_calibrations[0],
    )
    image_frame_indices = np.arange(len(all_centers), dtype=np.uint32)
    image_sensor_indices = np.zeros(len(all_centers), dtype=np.uint32)
    sensor_dimensions = np.array([[1280.0, 960.0]], dtype=np.float32)
    initial_points = expected_points + rng.normal(
        0.0, 0.05, expected_points.shape
    ).astype(np.float32)
    initial_point_error = np.linalg.norm(initial_points - expected_points)
    initial_pose_error = np.linalg.norm(
        rigs_from_world[2].tgt_origin_in_src() - section_centers[2]
    )

    def source(point_start, point_stop):
        point_count = point_stop - point_start
        observation_xy = np.stack(
            [
                camera.img_from_cam(
                    expected_points[point_start:point_stop] - center
                )
                for center in section_centers
            ],
            axis=1,
        ).reshape(-1, 2)
        return pycolmap.CasparRowTrackSource(
            np.ascontiguousarray(initial_points[point_start:point_stop]),
            np.arange(point_count, dtype=np.int64),
            np.arange(point_start, point_stop, dtype=np.uint32),
            np.arange(
                0,
                (point_count + 1) * len(section_centers),
                len(section_centers),
                dtype=np.int64,
            ),
            np.tile(
                np.arange(len(section_centers), dtype=np.int32), point_count
            ),
            np.ascontiguousarray(observation_xy, dtype=np.float32),
            (
                np.array([0], dtype=np.uint32)
                if point_start == 0
                else np.empty(0, dtype=np.uint32)
            ),
            np.arange(len(section_centers), dtype=np.uint32),
        )

    options = pycolmap.CasparBundleAdjustmentOptions()
    options.gpu_index = "0"
    options.solver_iter_max = 100
    row_points = np.ascontiguousarray(initial_points.copy())
    sources = (source(0, 24), source(24, 48))
    point_track_offsets = np.arange(len(expected_points) + 1, dtype=np.uint32)
    point_track_indices = np.arange(len(expected_points), dtype=np.uint32)
    source_support = np.ones(len(expected_points), dtype=np.uint16)
    active_frames = np.array([False, True, True, True, False, False, False])
    stats = pycolmap.caspar_row_section_stats(
        sources,
        point_track_offsets,
        point_track_indices,
        source_support,
        image_frame_indices,
        image_sensor_indices,
        sensor_dimensions,
        active_frames,
        1,
        np.array((0, 48), dtype=np.uint32),
    )
    result = pycolmap.caspar_refine_row_section(
        sources,
        point_track_offsets,
        point_track_indices,
        source_support,
        row_points,
        rigs_from_world,
        image_frame_indices,
        image_sensor_indices,
        sensors_from_rig,
        sensor_calibrations,
        sensor_dimensions,
        active_frames,
        1,
        48,
        options,
    )

    selected_stats = stats[-1]
    assert (
        selected_stats.point_count == result.point_count == len(expected_points)
    )
    assert selected_stats.observation_count == result.observation_count == 239
    assert (
        selected_stats.active_observation_count
        == result.active_observation_count
        == 144
    )
    np.testing.assert_array_equal(
        result.rigs_from_world[0].params, rigs_from_world[0].params
    )
    np.testing.assert_array_equal(
        result.rigs_from_world[4].params, rigs_from_world[4].params
    )
    np.testing.assert_array_equal(
        result.rigs_from_world[5].params, rigs_from_world[5].params
    )
    np.testing.assert_array_equal(
        result.rigs_from_world[6].params, rigs_from_world[6].params
    )
    assert result.summary.is_solution_usable()
    assert result.summary.final_score < result.summary.initial_score
    assert (
        np.linalg.norm(
            result.rigs_from_world[2].tgt_origin_in_src() - section_centers[2]
        )
        < initial_pose_error
    )
    assert np.linalg.norm(row_points - expected_points) < initial_point_error
