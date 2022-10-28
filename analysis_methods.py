from typing import List, Optional, Tuple
import cv2
import numpy as np

from utils import rotate_image

# color bounds for the colors blue and red respectively in the HSV color scheme
HSV_BR_COLOR_BOUNDS = [[(90, 25, 100), (120, 255, 255)], [(150, 50, 0), (179, 255, 255)]]

# location of the cascade classfier
CASCADE_CLASSIFIER = "cascade.xml"

class ImageContainer:
    """
    A class saving info about an image and its various forms used in further analysis.

    :param src: the source image
    """

    def __init__(self, src: cv2.Mat) -> None:
        self.src = src
        self.grayscale: cv2.Mat = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        self.hsv: cv2.Mat = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        self.ycrcb: cv2.Mat = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
        self.width: int = src.shape[0]
        self.height: int = src.shape[1]

def stamp_like_features(image_container: ImageContainer, x: int, y: int, width: int, height: int, threshold: Optional[cv2.Mat] = None, rotated_rect: Optional[cv2.Mat] = None, threshold_lower_bound: int = 140, threshold_upper_bound: int = 255, density_lower_bound: float = 0.064, density_upper_bound: float = 0.5, wh_ratio_lower_bound: float = 0.3, wh_ratio_upper_bound: float = 4.0, dimensions_lower_bound_factors: Tuple[int, int] = (20,50), dimensions_upper_bound_factors: Tuple[int, int] = (3,4)) -> Tuple[bool, List[int]]:
    """
    Given an area of an image find out if it has stamp like features. The following features are determined:

        - density of pixels
        - width to height ratio
        - lower bound on the dimensions relative to the size of the document
        - upper bound on the dimensions relative to the size of the document

    :param image_container: contains data about image and its various forms
    :param x: x coordinate of the left segment of the upright bounding rectangle
    :param y: y coordinate of the upper segment of the upright bounding rectangle
    :param width: width of the upright bounding rectangle
    :param height: height of the upright bounding rectangle
    :param threshold: threshold to use while determining the pixel density
    :param rotated_rect: a tuple (center, (width, height), angle) determining the minimal are bounding rectangle
    around the object
    :param threshold_lower_bound: the default thresholding lower bound when the thresholded image is not supplied
    :param threshold_upper_bound: the default thresholding upper bound when the thresholded image is not supplied
    :param density_lower_bound: the default pixel density lower bound
    :param density_upper_bound: the default pixel density upper bound
    :param wh_ratio_lower_bound: the default width to height lower bound
    :param wh_ratio_upper_bound: the default width to height upper bound
    :param dimensions_lower_bound_factors: a tuple of integers for width and height respectively, which dictate the lower bounds for dimensions of the detected object relative to the size of the entire image
    :param dimensions_upper_bound_factors: a tuple of integers for width and height respectively, which dictate the upper bounds for dimensions of the detected object relative to the size of the entire image

    :returns: a tuple (flag, corners) where flag is true iff the tracked features satisfy the defined bounds and
    corners is the corner coordinates of the detected object
    """
    # if a threshold is not given we create one, this is the case in non-color types of methods, i.e. cascade
    # classifier and Hough Circle Transform
    if threshold is None:
        _, threshold = cv2.threshold(
            255 - image_container.grayscale, threshold_lower_bound, threshold_upper_bound, cv2.THRESH_BINARY)

    # if a more fitting rotated rectangle around an object is found we calculate the density of the rotated
    # rectangle to correctly reflect the pixel density of the object, as rotating is resource sensitive we do
    # this only if the angle of rotation is high enough
    if rotated_rect is not None:
        center, wh, angle = rotated_rect
        if 10 < angle < 80:
            rotated_object = rotate_image(
                threshold, center, angle, int(wh[0]), int(wh[1]))
            density_in_bound = density_lower_bound < rotated_object.mean() / 255 <= density_upper_bound
        # if the rectangle is only slightly rotated we calculate the density just based on the (upright) bounding
        # rectangle
        else:
            density_in_bound = density_lower_bound < threshold[y:y +
                                                 height, x:x + width].mean() / 255 <= density_upper_bound
    # if the rotated rectangle is not supplied we calculate the density just based on the (upright) bounding
    # rectangle
    else:
        density_in_bound = density_lower_bound < threshold[y:y +
                                             height, x:x + width].mean() / 255 <= density_upper_bound

    # the width:height ratio is determined
    wh_ratio_in_bound = wh_ratio_lower_bound < width / height <= wh_ratio_upper_bound

    # the relative dimensions should follow these rules
    dimensions_lower_bound = width > image_container.width / dimensions_lower_bound_factors[0] and height > image_container.height / dimensions_lower_bound_factors[1]
    dimensions_upper_bound = width < image_container.width / dimensions_upper_bound_factors[0] and height < image_container.height / dimensions_upper_bound_factors[1]

    # verify the features
    flag = density_in_bound and wh_ratio_in_bound and dimensions_lower_bound and dimensions_upper_bound

    # save the corner of the (upright) bounding rectangle of the object
    corners = [x, y, x + width, y + height]

    # return whether the area has stamp like features and the corners of the bounding box around the object
    return flag, corners

def find_contours(image_container: ImageContainer, threshold: cv2.Mat, opening_kernel_dimensions:Tuple[int, int] = (4,4), closing_kernel_dimensions: Tuple[int, int] = (50,50)) -> List[List[int]]:
    """
    Find contours in a binary image (threshold) and verify if the contours have stamp like features and save the
    coordinates of verified objects.

    :param image_container: contains data about image and its various forms
    :threshold: threshold used for contour detection
    :opening_kernel_dimension: dimensions used for the opening kernel
    :closing_kernel_dimension: dimensions used for the closing kernel

    :returns: coordinates of contours with stamp like features
    """
    # first we perform some morphology operations on threshold to prepare the image for contour detection,
    # see https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

    # we use opening to get rid of noise
    opening_kernel = np.ones(opening_kernel_dimensions, np.uint8)
    opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, opening_kernel)

    # we closed open regions to find contours more easily
    closing_kernel = np.ones(closing_kernel_dimensions, np.uint8)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, closing_kernel)

    # find the contours in the closed image
    contours = cv2.findContours(
        closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    object_coordinates = []

    for c in contours:
        # verifying if the contour has the desired features
        stamp_like, corner_coordinates = stamp_like_features(image_container, *cv2.boundingRect(c), threshold=opening,
                                                                    rotated_rect=cv2.minAreaRect(c))
        # if yes we mark the object
        if stamp_like:
            object_coordinates.append(corner_coordinates)

    # all new coordinates are returned
    return object_coordinates


def find_circles(image_container: ImageContainer, hough_param1: int = 120, hough_param2: int = 50, min_radius: int = 100, max_radius: int = 200) -> List[List[int]]:
    """
    It is common to have stamps which have round shape. The Hough Circle Transform is used to detect circles and
    then the region is verified for stamp like features.

    See https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html for more info.

    :param image_container: contains data about image and its various forms
    :param hough_param1: upper threshold for the internal Canny edge detector
    :param hough_param2: threshold for center detection
    :param min_radius: minimum radius to be detected
    :param max_radius: maximum radius to be detected

    :returns: coordinates of round objects with stamp like features
    """
    # it is advisable to blur the image before applying the Hough Circle Transform
    blur = cv2.medianBlur(image_container.grayscale, 3)

    # use the algorithm
    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, image_container.height / 3,
                               param1=hough_param1, param2=hough_param2, minRadius=min_radius, maxRadius=max_radius)
    object_coordinates = []

    if circles is not None:
        circles = np.uint16(np.around(circles))

        for i in circles[0, :]:
            # find the bounding rectangle of circles
            x, y, r = i
            x = x - r
            y = y - r
            width = height = 2 * r

            # verify the region for desired features
            stamp_like, corner_coordinates = stamp_like_features(image_container=image_container,
                x=x, y=y, width=width, height=height)

            if stamp_like:
                object_coordinates.append(corner_coordinates)

    # return the stamp like objects
    return object_coordinates

def find_colored_objects_ycrcb(image_container: ImageContainer) -> List[List[int]]:
    """
    The YCrCb color space is especially suitable for stamp detection, see [1].

    :param image_container: contains data about image and its various forms

    :returns: coordinates of colored objects with stamp like features
    """
    # we split the channels of the YCrCb space and will do an analysis on the blue-difference and red-difference
    # components
    ycrcb = cv2.split(image_container.ycrcb)
    object_coordinates = []

    for i in range(2):
        c = ycrcb[i + 1]

        # threshold on the color components
        _, thresh = cv2.threshold(c, 140, 255, cv2.THRESH_BINARY)

        # find contours in the threshold
        object_coordinates += find_contours(image_container, thresh)
    
    return object_coordinates

def find_colored_objects_hsv(image_container: ImageContainer, color_lower_bound: int, color_upper_bound: int) -> List[List[int]]:
    """
    The HSV color space is suitable to detect regions of the image having a certain color shade, if the bounds
    are correctly chosen and tight enough this analysis yields good results.

    See https://en.wikipedia.org/wiki/HSL_and_HSV for info on HSV.

    :param image_container: contains data about image and its various forms
    :param color_lower_bound: HSV triple which acts as a lower bound for pixel color detection
    :param color_upper_bound: HSV triple which acts as an upper bound for pixel color detection

    :returns: coordinates of colored objects with stamp like features
    """
    # threshold the image with highlighted pixel with color in bounds
    thresh = cv2.inRange(image_container.hsv, color_lower_bound, color_upper_bound)

    # find contours in the threshold and return stamp like objects
    return find_contours(image_container, thresh)

def find_br_objects_hsv(image_container: ImageContainer) -> List[List[int]]:
    """
    Find blue and red objects in the image using the HSV color scheme.

    :param image_container: contains data about image and its various forms

    :returns: coordinates of colored objects with stamp like features
    """
    object_coordinates = []

    for bounds in HSV_BR_COLOR_BOUNDS:
            object_coordinates += find_colored_objects_hsv(image_container, *bounds)

    return object_coordinates

def cascade_classifier(image_container: ImageContainer, width_lower_bound: int=680, width_upper_bound:int = 910, height_lower_bound:int = 450, height_upper_bound:int = 500) -> List[List[int]]:
    """
    A custom trained cascade classifier is used to detect stamps in documents. The success rate of this method is
    limited and best results are for detected objects of width in range (680, 910) and height in range (450,
    500). The classifier was trained on the StaVer dataset: http://madm.dfki.de/downloads-ds-staver gathered
    during the research of [1].

    :param image_container: contains data about image and its various forms
    :width_lower_bound: the width lower bound for the detected object
    :width_upper_bound: the width upper bound for the detected object
    :height_lower_bound: the height lower bound for the detected object
    :height_upper_bound: the height upper bound for the detected object

    :returns: coordinates of objects marked as a stamp using the custom cascade classifier
    """

    # initialize the custom classifier
    classifier = cv2.CascadeClassifier(CASCADE_CLASSIFIER)

    # detect objects
    detected_objects = classifier.detectMultiScale(
        image_container.grayscale, minSize=(width_lower_bound, height_lower_bound))

    object_coordinates = []

    if len(detected_objects) != 0:
        for (x, y, width, height) in detected_objects:
            # make sure the detected objects have dimensions within bounds
            if width_lower_bound < width < width_upper_bound and height_lower_bound < height < height_upper_bound:
                object_coordinates.append([x, y, x + width, y + height])

    # return detected objects
    return object_coordinates
