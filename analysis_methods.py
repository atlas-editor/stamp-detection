from dataclasses import dataclass
from typing import List, Tuple
import cv2
import numpy as np
from stamp_feature_analysis import StampFeatureSettings, stamp_like_features

from utils import create_binary_image, rotate_image

@dataclass
class AnalysisSettings:
    """
    A class saving all the parameter values and settings for the analysis.

    :param opening_kernel_dimensions: dimensions used for the opening kernel in contour detection
    :param closing_kernel_dimensions: dimensions used for the closing kernel in contour detection
    :param hough_dp: inverse ratio of the accumulator resolution to the image resolution when using the Hough transform
    :param hough_min_distance_factor: minimum distance between the centers of the detected circles when using the Hough transform as the fraction of the height of the entire document
    :param hough_param1: upper threshold for the internal Canny edge detector used in circle detection via Hough transform
    :param hough_param2: threshold for center detection used in circle detection via Hough transform
    :param hough_min_radius: minimum radius to be detected used in circle detection via Hough transform
    :param hough_max_radius: maximum radius to be detected used in circle detection via Hough transform
    :param hsv_br_color_bounds: color bounds for the colors blue and red respectively in the HSV color scheme
    :param cascade_classifier_path: location of the cascade classifier
    :param cascade_classifier_width_lower_bound: the width lower bound for the detected object when using the cascade classifier
    :param cascade_classifier_width_upper_bound: the width upper bound for the detected object when using the cascade classifier
    :param cascade_classifier_height_lower_bound: the height lower bound for the detected object when using the cascade classifier
    :param cascade_classifier_height_upper_bound: the height upper bound for the detected object when using the cascade classifier
    :param stamp_feature_settings: defining parameters for objects with stamp like features, see StampFeatureSettings for more info
    """
    contour_opening_kernel_dimensions: Tuple[int, int] = (4, 4)
    contour_closing_kernel_dimensions: Tuple[int, int] = (50, 50)
    hough_dp: int = 2
    hough_min_distance_factor: int = 3 
    hough_param1: int = 120
    hough_param2: int = 50
    hough_min_circle_radius: int = 100
    hough_max_circle_radius: int = 200
    hsv_br_color_bounds: Tuple[Tuple, Tuple] = (((90, 25, 100), (120, 255, 255)), ((150, 50, 0), (179, 255, 255)))
    cascade_classifier_path: str = "./cascade.xml"
    cascade_classifier_width_lower_bound: int = 680
    cascade_classifier_width_upper_bound: int = 1000
    cascade_classifier_height_lower_bound: int = 450
    cascade_classifier_height_upper_bound: int = 550
    stamp_feature_settings: StampFeatureSettings = StampFeatureSettings()

@dataclass
class ImageContainer:
    """
    A class saving info about an image and its various forms used in further analysis.

    :param source_image: the source image
    """
    source_image: cv2.Mat

    def __post_init__(self) -> None:
        self.grayscale: cv2.Mat = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2GRAY)
        self.hsv: cv2.Mat = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2HSV)
        self.ycrcb: cv2.Mat = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2YCrCb)
        self.width: int = self.source_image.shape[0]
        self.height: int = self.source_image.shape[1]

def find_circles(image_container: ImageContainer, analysis_settings: AnalysisSettings) -> List[List[int]]:
    """
    It is common to have stamps which have round shape. The Hough Circle Transform is used to detect circles and
    then the region is verified for stamp like features.

    See https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html for more info.

    :param image_container: contains data about image and its various forms
    :param analysis_settings: container with parameters, see AnalysisSettings for more info

    :returns: coordinates of round objects with stamp like features
    """
    # it is advisable to blur the image before applying the Hough Circle Transform
    blur = cv2.medianBlur(image_container.grayscale, 3)

    # use the algorithm
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, analysis_settings.hough_dp, image_container.height / analysis_settings.hough_min_distance_factor,
                               param1=analysis_settings.hough_param1, param2=analysis_settings.hough_param2, minRadius=analysis_settings.hough_min_circle_radius, maxRadius=analysis_settings.hough_max_circle_radius)
    object_coordinates = []

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # find the bounding rectangle of circles
            x, y, r = i
            x = x - r
            y = y - r
            width = height = 2 * r

            stamp_grayscale = image_container.grayscale[y : y + height, x : x + width]
            stamp_candidate = create_binary_image(stamp_grayscale)

            # verify the region for desired features
            stamp_like = stamp_like_features(stamp_candidate, document_width=image_container.width, document_height=image_container.height, feature_settings=analysis_settings.stamp_feature_settings)
            if stamp_like:
                corner_coordinates = [x, y, x + width, y + height]
                object_coordinates.append(corner_coordinates)

    # return the stamp like objects
    return object_coordinates

def find_colored_objects_ycrcb(image_container: ImageContainer, analysis_settings: AnalysisSettings) -> List[List[int]]:
    """
    The YCrCb color space is especially suitable for stamp detection, see [1].

    :param image_container: contains data about image and its various forms
    :param analysis_settings: container with parameters, see AnalysisSettings for more info

    :returns: coordinates of colored objects with stamp like features
    """
    # we split the channels of the YCrCb space and will do an analysis on the blue-difference and red-difference
    # components
    ycrcb = cv2.split(image_container.ycrcb)
    object_coordinates = []

    for i in range(2):
        color_component = ycrcb[i + 1]

        # threshold on the color components
        threshold = create_binary_image(color_component)

        # find contours in the threshold
        object_coordinates += _find_contours(image_container, threshold, analysis_settings)

    return object_coordinates

def find_br_objects_hsv(image_container: ImageContainer, analysis_settings: AnalysisSettings) -> List[List[int]]:
    """
    Find blue and red objects in the image using the HSV color scheme.

    :param image_container: contains data about image and its various forms
    :param analysis_settings: container with parameters, see AnalysisSettings for more info

    :returns: coordinates of colored objects with stamp like features
    """
    object_coordinates = []

    for bounds in analysis_settings.hsv_br_color_bounds:
        object_coordinates += _find_colored_objects_hsv(image_container, *bounds, analysis_settings)

    return object_coordinates

def cascade_classifier(image_container: ImageContainer, analysis_settings: AnalysisSettings) -> List[List[int]]:
    """
    A custom trained cascade classifier is used to detect stamps in documents. The success rate of this method is
    limited and best results are for detected objects of width in range (680, 1000) and height in range (450,
    550). The classifier was trained on the StaVer dataset: http://madm.dfki.de/downloads-ds-staver gathered
    during the research of [1].

    :param image_container: contains data about image and its various forms
    :param analysis_settings: container with parameters, see AnalysisSettings for more info

    :returns: coordinates of objects marked as a stamp using the custom cascade classifier
    """

    # initialize the custom classifier
    classifier = cv2.CascadeClassifier(analysis_settings.cascade_classifier_path)

    # detect objects
    detected_objects = classifier.detectMultiScale(
        image_container.grayscale, minSize=(analysis_settings.cascade_classifier_width_lower_bound, analysis_settings.cascade_classifier_height_lower_bound))

    object_coordinates = []

    if len(detected_objects) != 0:
        for (x, y, width, height) in detected_objects:
            # make sure the detected objects have dimensions within bounds
            if analysis_settings.cascade_classifier_width_lower_bound < width < analysis_settings.cascade_classifier_width_upper_bound and analysis_settings.cascade_classifier_height_lower_bound < height < analysis_settings.cascade_classifier_height_upper_bound:
                object_coordinates.append([x, y, x + width, y + height])

    # return detected objects
    return object_coordinates


def _find_contours(image_container: ImageContainer, threshold: cv2.Mat, analysis_settings: AnalysisSettings) -> List[List[int]]:
    """
    Find contours in a binary image (threshold) and verify if the contours have stamp like features and save the
    coordinates of verified objects.

    :param image_container: contains data about image and its various forms
    :param threshold: threshold used for contour detection
    :param analysis_settings: container with parameters, see AnalysisSettings for more info

    :returns: coordinates of contours with stamp like features
    """
    # first we perform some morphology operations on threshold to prepare the image for contour detection,
    # see https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

    # we use opening to get rid of noise
    opening_kernel = np.ones(analysis_settings.contour_opening_kernel_dimensions, np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, opening_kernel)

    # we closed open regions to find contours more easily
    closing_kernel = np.ones(analysis_settings.contour_closing_kernel_dimensions, np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, closing_kernel)

    # find the contours in the closed image
    contours = cv2.findContours(
        closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    object_coordinates = []

    for c in contours:
        # verifying if the contour has the desired features

        # first find the min area rectangle around an object (usually rotated) is found, as rotating is resource sensitive we do
        # it only if the angle of rotation is high enough, define also the upright bounding rectangle
        center, wh, angle = cv2.minAreaRect(c)
        x, y, width, height = cv2.boundingRect(c)
        if 10 < angle < 80:
            stamp_candidate = rotate_image(opening, center, angle, int(wh[0]), int(wh[1]))
        # else just use the upright bounding rectangle
        else:
            stamp_candidate = opening[y : y + height, x : x + width]
        stamp_like = stamp_like_features(stamp_candidate, document_width=image_container.width, document_height=image_container.height, feature_settings=analysis_settings.stamp_feature_settings)
        # if yes we mark the object      
        if stamp_like:
            corner_coordinates = [x, y, x + width, y + height]
            object_coordinates.append(corner_coordinates)

    # all new coordinates are returned
    return object_coordinates


def _find_colored_objects_hsv(image_container: ImageContainer, color_lower_bound: int, color_upper_bound: int, analysis_settings: AnalysisSettings) -> \
        List[List[int]]:
    """
    The HSV color space is suitable to detect regions of the image having a certain color shade, if the bounds
    are correctly chosen and tight enough this analysis yields good results.

    See https://en.wikipedia.org/wiki/HSL_and_HSV for info on HSV.

    :param image_container: contains data about image and its various forms
    :param color_lower_bound: HSV triple which acts as a lower bound for pixel color detection
    :param color_upper_bound: HSV triple which acts as an upper bound for pixel color detection
    :param analysis_settings: container with parameters, see AnalysisSettings for more info

    :returns: coordinates of colored objects with stamp like features
    """
    # threshold the image with highlighted pixel with color in bounds
    threshold = cv2.inRange(image_container.hsv, color_lower_bound, color_upper_bound)

    # find contours in the threshold and return stamp like objects
    return _find_contours(image_container, threshold, analysis_settings)