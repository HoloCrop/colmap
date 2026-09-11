import numpy as np
import pytest
import pycolmap

cv2 = pytest.importorskip("cv2")


@pytest.mark.parametrize("count", [0, 4, 5, 8, 14])
def test_source_coordinate_restoration(count):
    # Mixed cameras and pixels near/beyond image boundaries exercise dispatch,
    # resize/intrinsic conversion and the thin-prism/tilted sensor convention.
    rng = np.random.default_rng(42)
    pixels = rng.uniform([-100, -100], [2000, 1400], size=(1000, 2)).astype(np.float32)
    indices = rng.integers(0, 2, size=len(pixels), dtype=np.int32)
    coefficients = np.array([.1, -.02, .003, -.005, .002, .01, -.003, .001,
                             .002, -.001, .003, -.002, .04, -.03])[:count]
    prepared = np.array([1300, 1200, 900, 600.])
    source = np.array([1700, 1600, 1000, 700.])
    cameras = [pycolmap.SourceCoordinateCamera(prepared, source, coefficients),
               pycolmap.SourceCoordinateCamera(source, prepared, -coefficients)]
    result = pycolmap.restore_source_coordinates(cameras, indices, pixels)
    for index, (before, after, distortion) in enumerate(
        [(prepared, source, coefficients), (source, prepared, -coefficients)]
    ):
        selected = indices == index
        normalized = (pixels[selected].astype(np.float64) - before[2:]) / before[:2]
        points = np.column_stack((normalized, np.ones(len(normalized))))
        intrinsic = np.array([[after[0], 0, after[2]], [0, after[1], after[3]], [0, 0, 1]])
        expected, _ = cv2.projectPoints(points, np.zeros(3), np.zeros(3), intrinsic, distortion)
        np.testing.assert_allclose(result[selected], expected.reshape(-1, 2), rtol=0, atol=2e-4)


def test_source_coordinate_restoration_empty():
    result = pycolmap.restore_source_coordinates([], np.empty(0, np.int32), np.empty((0, 2), np.float32))
    assert result.shape == (0, 2)
