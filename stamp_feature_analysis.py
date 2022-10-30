from dataclasses import dataclass
from typing import Tuple
import cv2

@dataclass(frozen=True)
class StampFeatureSettings:
    """
    A class saving all the parameter values for stamp features.

    :param density_lower_bound: the default pixel density lower bound
    :param density_upper_bound: the default pixel density upper bound
    :param wh_ratio_lower_bound: the default width to height lower bound
    :param wh_ratio_upper_bound: the default width to height upper bound
    :param dimensions_lower_bound_factors: a tuple of integers for width and height respectively, which dictate the
    lower bounds for dimensions of the detected object relative to the size of the entire image
    :param dimensions_upper_bound_factors: a tuple of integers for width and height respectively, which dictate the
    upper bounds for dimensions of the detected object relative to the size of the entire image
    """
    density_lower_bound: float = 0.064
    density_upper_bound: float = 0.5
    wh_ratio_lower_bound: float = 0.3
    wh_ratio_upper_bound: float = 4.0
    dimensions_lower_bound_factors: Tuple[int, int] = (20, 50)
    dimensions_upper_bound_factors: Tuple[int, int] = (3, 4)


def stamp_like_features(stamp: cv2.Mat, document_width: int, document_height: int, feature_settings: StampFeatureSettings) -> bool:
    """
    Given an area of an image find out if it has stamp like features. The following features are determined:

        - density of pixels
        - width to height ratio
        - lower bound on the dimensions relative to the size of the document
        - upper bound on the dimensions relative to the size of the document

    :param stamp: contains part of the document which is checked for stamp like features
    :param document_width: width of the entire document
    :param document_height: height of the entire document
    :param feature_settings: container with parameters, see StampFeatureSettings for more info

    :returns: a boolean flag which is true iff the tracked features satisfy the defined bounds
    """
    stamp_width = stamp.shape[0]
    stamp_height = stamp.shape[1]

    # we calculate the density
    density_check = density_in_bound(stamp, feature_settings.density_lower_bound, feature_settings.density_upper_bound)

    # the width:height ratio is determined
    wh_ratio_check = wh_ratio_in_bound(stamp_width, stamp_height, feature_settings.wh_ratio_lower_bound, feature_settings.wh_ratio_upper_bound)

    # the relative dimensions should follow the given rules
    dimensions_check = dimension_in_bound(stamp_width, stamp_height, document_width, document_height, feature_settings.dimensions_lower_bound_factors, feature_settings.dimensions_upper_bound_factors)

    # return whether the area has stamp like features
    return density_check and wh_ratio_check and dimensions_check


def density_in_bound(stamp: cv2.Mat, density_lower_bound: float, density_upper_bound: float) -> bool:
    """
    Given an area of a binary image find out if its density falls in the defined bounds.

    :param stamp: contains part of the document which is checked
    :param density_lower_bound: pixel density lower bound
    :param density_upper_bound: pixel density upper bound

    :returns: a boolean flag which is true iff the pixel density falls in the defined bounds.
    """
    if stamp.shape[0] == 0 or stamp.shape[1] == 0:
        return False
    return density_lower_bound <= (stamp.mean()/255) <= density_upper_bound

def wh_ratio_in_bound(width: int, height: int, wh_ratio_lower_bound: float, wh_ratio_upper_bound: float) -> bool:
    """
    Given the width and height of an area of an image find out if its width to height ratio falls in the defined bounds.

    :param width: width of the object
    :param height: height of the object
    :param wh_ratio_lower_bound: width to height lower bound
    :param wh_ratio_upper_bound: width to height upper bound

    :returns: a boolean flag which is true iff the width to height ratio falls in the defined bounds
    """
    return wh_ratio_lower_bound <= width / height <= wh_ratio_upper_bound

def dimension_in_bound(obj_width: int, obj_height: int, img_width: int, img_height: int, dimensions_lower_bound_factors: Tuple[int, int], dimensions_upper_bound_factors: Tuple[int, int]) -> bool:
    """
    Given an area of a binary image find out if its width to height ratio falls in the defined bounds.

    :param stamp: contains part of the document which is checked
    :param wh_ratio_lower_bound: width to height lower bound
    :param wh_ratio_upper_bound: width to height upper bound

    :returns: a boolean flag which is true iff the width to height ratio falls in the defined bounds
    """    
    lower_bound = obj_width >= img_width / dimensions_lower_bound_factors[0] and obj_height >= img_height / dimensions_lower_bound_factors[1]
    upper_bound = obj_width <= img_width / dimensions_upper_bound_factors[0] and obj_height <= img_height / dimensions_upper_bound_factors[1]

    return lower_bound and upper_bound
