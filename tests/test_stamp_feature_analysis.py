
import numpy as np
from stamp_feature_analysis import StampFeatureSettings, density_in_bound, dimension_in_bound, stamp_like_features, wh_ratio_in_bound

def test_density_in_bound() -> None:
    # "nonstamp" pixels
    unfilled = np.zeros([40, 20, 3], dtype = np.uint8)
    # "stamp" pixels
    filled = 255*np.ones([10, 20, 3], dtype = np.uint8)
    # stack them together, note the "stamp" pixels are exactly 20%
    stamp = np.vstack((unfilled, filled))
    density_lb = 0.1
    density_ub = 0.4

    assert density_in_bound(stamp, density_lb, density_ub) == True

def test_density_out_bound() -> None:
    unfilled = np.zeros([49, 20, 3], dtype = np.uint8)
    filled = 255*np.ones([1, 20, 3], dtype = np.uint8)
    # stack together, "stamp" pixels are 2%
    stamp = np.vstack((unfilled, filled))
    density_lb = 0.1
    density_ub = 0.4

    assert density_in_bound(stamp, density_lb, density_ub) == False

def test_wh_ratio_in_bound() -> None:
    stamp_width = 50
    stamp_height = 20

    wh_ratio_lb = 0.5
    wh_ratio_ub = 3

    assert wh_ratio_in_bound(stamp_width, stamp_height, wh_ratio_lb, wh_ratio_ub) == True

def test_wh_ratio_out_bound() -> None:
    stamp_width = 50
    stamp_height = 20

    wh_ratio_lb = 0.5
    wh_ratio_ub = 2

    assert wh_ratio_in_bound(stamp_width, stamp_height, wh_ratio_lb, wh_ratio_ub) == False

def test_dimension_in_bound() -> None:
    stamp_width = 50
    stamp_height = 20

    img_width = 500
    img_height = 1000

    dimension_lower_bound_factors = (10, 100)
    dimensions_upper_bound_factors = (2,2)

    assert dimension_in_bound(stamp_width, stamp_height, img_width, img_height, dimension_lower_bound_factors, dimensions_upper_bound_factors) == True

def test_dimension_out_bound() -> None:
    stamp_width = 50
    stamp_height = 20

    img_width = 500
    img_height = 1000

    dimension_lower_bound_factors = (10, 10)
    dimensions_upper_bound_factors = (2,2)

    assert dimension_in_bound(stamp_width, stamp_height, img_width, img_height, dimension_lower_bound_factors, dimensions_upper_bound_factors) == False

def test_stamp_like_features() -> None:
    parameters = StampFeatureSettings(0.1, 0.3, 1, 5, (20,50), (5,10))

    unfilled = np.zeros([40, 20, 3], dtype = np.uint8)
    filled = 255*np.ones([10, 20, 3], dtype = np.uint8)
    stamp = np.vstack((unfilled, filled))

    doc_width = 500
    doc_height = 1000

    assert stamp_like_features(stamp, doc_width, doc_height, parameters) == True

def test_stamp_unlike_features() -> None:
    parameters = StampFeatureSettings(0.1, 0.11, 1, 5, (20,50), (5,10))

    unfilled = np.zeros([40, 20, 3], dtype = np.uint8)
    filled = 255*np.ones([10, 20, 3], dtype = np.uint8)
    stamp = np.vstack((unfilled, filled))

    doc_width = 500
    doc_height = 1000

    assert stamp_like_features(stamp, doc_width, doc_height, parameters) == False