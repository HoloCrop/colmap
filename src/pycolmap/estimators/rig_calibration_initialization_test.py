from itertools import combinations

import numpy as np

import pycolmap


def test_initialization_excludes_heldout_geometry_and_preserves_pair_direction(
    tmp_path,
):
    pycolmap.set_random_seed(12)
    reconstruction = pycolmap.synthesize_dataset(
        pycolmap.SyntheticDatasetOptions(
            num_rigs=1,
            num_cameras_per_rig=3,
            num_frames_per_rig=3,
            num_points3D=0,
            num_points2D_without_point3D=0,
        )
    )
    rig = next(iter(reconstruction.rigs.values()))
    camera_ids = np.asarray(sorted(reconstruction.cameras), dtype=np.uint32)
    sensors = [
        pycolmap.sensor_t(pycolmap.SensorType.CAMERA, int(camera_id))
        for camera_id in camera_ids
    ]
    expected_cameras = [
        pycolmap.Rigid3d()
        if rig.is_ref_sensor(sensor)
        else pycolmap.Rigid3d(rig.sensor_from_rig(sensor).matrix())
        for sensor in sensors
    ]
    world_poses = [
        reconstruction.frame(frame_id).rig_from_world
        for frame_id in sorted(reconstruction.frames)
    ]
    frame_poses = [pose * world_poses[0].inverse() for pose in world_poses]
    # Deliberately interleave image IDs across cameras and frames so reading
    # canonical database pairs must invert several written relative poses.
    selected_images = np.asarray([[91, 7, 66], [5, 80, 12], [41, 2, 30]])
    locations = tuple(np.ndindex(selected_images.shape))
    moved_index = next(
        index
        for index, sensor in reversed(tuple(enumerate(sensors)))
        if not rig.is_ref_sensor(sensor)
    )
    expected = expected_cameras[moved_index]
    moved_rotation = pycolmap.Rotation3d(np.array([0.4, -0.2, 0.1]))
    moved_center = expected.tgt_origin_in_src() + np.array([0.08, -0.06, 0.04])
    moved_pose = pycolmap.Rigid3d(
        moved_rotation * expected.rotation,
        -(moved_rotation * expected.rotation * moved_center),
    )
    heldout_cameras = list(expected_cameras)
    heldout_cameras[moved_index] = moved_pose
    path = tmp_path / "initialization.db"
    with pycolmap.Database.open(path) as database:
        for camera in reconstruction.cameras.values():
            database.write_camera(camera, use_camera_id=True)
        for image_ids, cameras in (
            (selected_images, expected_cameras),
            (selected_images + 100, heldout_cameras),
        ):
            for frame_index, camera_index in locations:
                image_id = int(image_ids[frame_index, camera_index])
                database.write_image(
                    pycolmap.Image(
                        image_id=image_id,
                        camera_id=int(camera_ids[camera_index]),
                        name=f"{image_id}.jpg",
                    ),
                    use_image_id=True,
                )
            for (frame1, camera1), (frame2, camera2) in combinations(
                locations, 2
            ):
                pose1 = cameras[camera1] * frame_poses[frame1]
                pose2 = cameras[camera2] * frame_poses[frame2]
                relative = pose2 * pose1.inverse()
                relative.translation /= np.linalg.norm(relative.translation)
                database.write_two_view_geometry(
                    int(image_ids[frame1, camera1]),
                    int(image_ids[frame2, camera2]),
                    pycolmap.TwoViewGeometry(
                        config=pycolmap.TwoViewGeometryConfiguration.CALIBRATED,
                        cam2_from_cam1=relative,
                    ),
                )

    rig.set_sensor_from_rig(sensors[moved_index], moved_pose)
    distances = np.asarray(
        [np.linalg.norm(frame_poses[-1].tgt_origin_in_src())]
    )
    original_distances = distances.copy()
    options = pycolmap.RigCalibrationOptions()
    options.ceres.solver_options.num_threads = 1
    options.ceres.solver_options.max_num_iterations = 100
    result = pycolmap.initialize_rig_calibration(
        options=options,
        rig_id=rig.rig_id,
        reconstruction=reconstruction,
        database_path=path,
        image_ids=selected_images[None].astype(np.uint32),
        camera_ids=camera_ids,
        rigs_from_group=np.asarray([[pose.matrix() for pose in frame_poses]]),
        first_to_last_distances=distances,
        distance_stddev=0.01,
    )

    assert result.num_pairs == len(locations) * (len(locations) - 1) // 2
    actual = rig.sensor_from_rig(sensors[moved_index])
    np.testing.assert_allclose(
        actual.tgt_origin_in_src(), expected.tgt_origin_in_src(), atol=1e-4
    )
    np.testing.assert_allclose(
        actual.rotation.matrix(), expected.rotation.matrix(), atol=1e-4
    )
    np.testing.assert_array_equal(distances, original_distances)
