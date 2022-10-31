from typing import List, Tuple
import cv2
import numpy as np
from analysis_methods import AnalysisSettings, ImageContainer, find_br_objects_hsv, find_circles, \
    find_colored_objects_ycrcb
import stamp_feature_analysis


def _prepare_test(color=List[int]) -> Tuple[ImageContainer, List[int], AnalysisSettings]:
    """
    Prepare artificial image with one circle whose color is defined in the `color` variable in the BGR scheme.

    :param color: a triple defining the color of the circle in BGR scheme

    :returns: a triple with the image, the exact bounds around the circle and the necessary parameters for the analysis methods
    """
    parameters = AnalysisSettings(hough_min_circle_radius=3, hough_max_circle_radius=100,
                                  stamp_feature_settings=stamp_feature_analysis.StampFeatureSettings(
                                      density_upper_bound=1))

    bg = 255 * np.ones([500, 500, 3], dtype=np.uint8)

    circle = cv2.circle(bg, (50, 50), 20, color, -1)

    circle_bound = [30, 30, 70, 70]

    img = ImageContainer(circle)

    return img, circle_bound, parameters


def test_find_circles() -> None:
    """
    Using the artificial image from `_prepare_test` we test whether the circle is correctly detected.
    """
    # prepare the artificial document with black circle
    img, circle_bound, parameters = _prepare_test([0, 0, 0])

    # run the tested  method
    detected_circles = find_circles(img, parameters)

    # only one circle should be detected and should not be far off from the real values (we test for 10% error)
    assert len(detected_circles) == 1

    assert np.isclose(circle_bound, detected_circles[0], rtol=0.1).all()


def test_find_colored_objects_ycrcb() -> None:
    """
    Using the artificial image from `_prepare_test` we test whether the circle is correctly detected if colored blue.
    """
    # prepare the artificial document with blue circle
    img, circle_bound, parameters = _prepare_test([255, 0, 0])

    # run the tested  method
    objects = find_colored_objects_ycrcb(img, parameters)

    # only one circle should be detected and should not be far off from the real values (we test for 10% error)
    assert len(objects) == 1

    assert np.isclose(circle_bound, objects[0], rtol=0.1).all()


def test_find_br_objects_hsv() -> None:
    """
    Using the artificial image from `_prepare_test` we test whether the circle is correctly detected if colored blue.
    """
    # prepare the artificial document with blue circle
    img, circle_bound, parameters = _prepare_test([255, 0, 0])

    # run the tested  method
    objects = find_br_objects_hsv(img, parameters)

    # only one circle should be detected and should not be far off from the real values (we test for 10% error)
    assert len(objects) == 1

    assert np.isclose(circle_bound, objects[0], rtol=0.1).all()
