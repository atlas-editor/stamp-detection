"""
Provides a way of analyzing official documents to detect the presence of stamps.

Disclaimer: all constant are chosen empirically on a small set of documents to suit our purposes and may not work
well for domains outside of official documents.

References:

    [1] B. Micenkova and J. v. Beusekom, "Stamp Detection in Color Document Images", 2011
    [2] P. Forczmanski and A. Markiewicz, "Stamps Detection and Classification Using Simple Features Ensemble", 2015
"""

from typing import List
import cv2
import analysis_methods

from utils import remove_duplicates


class StampAnalyzer:
    """
    A class utilizing several methods to detect stamps in official documents.

    :param src: document on which to perform the analysis
    """

    def __init__(self, src: cv2.Mat) -> None:
        self.image_container = analysis_methods.ImageContainer(src)

    def analyze(self) -> List[List[int]]:
        """
        Function utilizing all the methods used to find stamps in official documents.

        The results (if any) are returned as a list of lists (of size 4) where each 4-tuple defines a rectangle
        around a detected stamp. The 4-tuple (x1, y1, x2, y2) determines a rectangle whose left segment has x
        coordinate x1, upper segment has y coordinate y1, right segment has x coordinate x2 and bottom segment has y
        coordinate y2.

        :returns: coordinates of objects classified as stamps by the analyzer
        """
        stamp_coordinates = []
        pipeline = [analysis_methods.find_circles, analysis_methods.find_colored_objects_ycrcb,
                    analysis_methods.find_br_objects_hsv, analysis_methods.cascade_classifier]

        for method in pipeline:
            stamp_coordinates += method(self.image_container)

        remove_duplicates(stamp_coordinates)

        return stamp_coordinates


if __name__ == '__main__':
    pass
