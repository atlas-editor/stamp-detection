"""
Provides a way of analyzing official documents to detect the presence of stamps.

Disclaimer: all constant are chosen empirically on a small set of documents to suit our purposes and may not work
well for domains outside of official documents.

References:

    [1] B. Micenkova and J. v. Beusekom, "Stamp Detection in Color Document Images", 2011
    [2] P. Forczmanski and A. Markiewicz, "Stamps Detection and Classification Using Simple Features Ensemble", 2015
"""

import itertools
import cv2
import argparse
import numpy as np


class StampAnalyzer:
    """
    A class utilizing several methods to detect stamps in official documents.

    :param src: document on which to perform the analysis
    """

    def __init__(self, src):
        self.src = src  # the source image
        self.output = src.copy()  # a copy of the image for showing the output
        # grascale version of the image
        self.grayscale = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # HSV version of the image used for finding colored objects
        self.hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        # YCrCb version of the image used for finding colored
        self.ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
        # objects
        self.height, self.width = src.shape[:2]  # the shape of the image
        self.stamp_coordinates = []  # a list where stamp coordinates will be saved

        # analyze the image
        self.analyze()

    def _rotate_image(self, image, center, theta, width, height):
        ''' 
        Rotates OpenCV image around center with angle theta (in deg) then crops the image according to width and height.

        Source: https://stackoverflow.com/a/11627903.

        :param image: image to rotate
        :param center: center of rotation
        :param theta: angle of rotation
        :param width: width of the region we are cropping
        :param height: height of the region we are cropping
        '''

        # Uncomment for theta in radians
        # theta *= 180/np.pi

        # cv2.warpAffine expects shape in (length, height)
        shape = (image.shape[1], image.shape[0])

        matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
        image = cv2.warpAffine(src=image, M=matrix, dsize=shape)

        x = int(center[0] - width / 2)
        y = int(center[1] - height / 2)

        image = image[y:y + height, x:x + width]

        return image

    def find_contours(self, threshold):
        """
        Find contours in a binary image (threshold) and verify if the contours have stamp like features and save the
        coordinates of verified objects.

        :threshold: threshold used for contour detection
        :returns: coordinates of contours with stamp like features
        """
        # first we perform some morphology operations on threshold to prepare the image for contour detection,
        # see https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

        # we use opening to get rid of noise
        opening_kernel = np.ones((4, 4), np.uint8)
        opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, opening_kernel)

        # we closed open regions to find contours more easily
        closing_kernel = np.ones((50, 50), np.uint8)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, closing_kernel)

        # find the contours in the closed image
        contours = cv2.findContours(
            closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]

        object_coordinates = []

        for c in contours:
            # verifying if the contour has the desired features
            stamp_like, corner_coordinates = self._stamp_like_features(*cv2.boundingRect(c), threshold=opening,
                                                                       rotated_rect=cv2.minAreaRect(c))
            # if yes we mark the object
            if stamp_like:
                object_coordinates.append(corner_coordinates)

        # all new coordinates are saved
        self.stamp_coordinates += object_coordinates
        return object_coordinates

    def find_colored_objects_ycbcr(self):
        """
        The YCrCb color space is especially suitable for stamp detection, see [1].
        """
        # we split the channels of the YCrCb space and will do an analysis on the blue-difference and red-difference
        # components
        ycrcb = cv2.split(self.ycrcb)

        for i in range(2):
            c = ycrcb[i + 1]

            # threshold on the color components
            _, thresh = cv2.threshold(c, 140, 255, cv2.THRESH_BINARY)

            # find contours in the threshold
            self.find_contours(thresh)

    def find_colored_objects_hsv(self, color_lower_bound, color_upper_bound):
        """
        The HSV color space is suitable to detect regions of the image having a certain color shade, if the bounds
        are correctly chosen and tight enough this analysis yields good results.

        See https://en.wikipedia.org/wiki/HSL_and_HSV for info on HSV.

        :param color_lower_bound: HSV triple which acts as a lower bound for pixel color detection
        :param color_upper_bound: HSV triple which acts as an upper bound for pixel color detection
        """
        # threshold the image with highlighted pixel with color in bounds
        thresh = cv2.inRange(self.hsv, color_lower_bound, color_upper_bound)

        # find contours in the threshold
        self.find_contours(thresh)

    def find_circles(self):
        """
        It is common to have stamps which have round shape. The Hough Circle Transform is used to detect circles and
        then the region is verified for stamp like features.

        See https://docs.opencv.org/3.4/d4/d70/tutorial_hough_circle.html for more info.

        :returns: coordinates of round objects with stamp like features
        """
        # it is advisable to blur the image before applying the Hough Circle Transform
        blur = cv2.medianBlur(self.grayscale, 3)

        # use the algorithm
        circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, self.height / 3,
                                   param1=120, param2=50, minRadius=100, maxRadius=200)
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
                stamp_like, corner_coordinates = self._stamp_like_features(
                    x=x, y=y, width=width, height=height)

                if stamp_like:
                    object_coordinates.append(corner_coordinates)

        # save the stamp like objects
        self.stamp_coordinates += object_coordinates
        return object_coordinates

    def cascade_classifier(self):
        """
        A custom trained cascade classifier is used to detect stamps in documents. The success rate of this method is
        limited and best results are for detected objects of width in range (680, 910) and height in range (450,
        500). The classifier was trained on the StaVer dataset: http://madm.dfki.de/downloads-ds-staver gathered
        during the research of [1].

        :returns: coordinates of objects marked as a stamp using the custom cascade classifier
        """

        # initialize the custom classifier
        classifier = cv2.CascadeClassifier("cascade.xml")

        # detect objects
        detected_objects = classifier.detectMultiScale(
            self.grayscale, minSize=(50, 50))

        object_coordinates = []

        if len(detected_objects) != 0:
            for (x, y, width, height) in detected_objects:
                # detected objects with dimensions outside of these bounds have very limited success rate
                if 680 < width < 910 and 450 < height < 500:
                    object_coordinates.append([x, y, x + width, y + height])

        # save detected objects
        self.stamp_coordinates += object_coordinates
        return object_coordinates

    def _stamp_like_features(self, x, y, width, height, threshold=None, rotated_rect=None):
        """
        Given an area of the image find out if it has stamp like features. The following features are determined:

            - density of pixels
            - width to height ratio
            - lower bound on the dimensions relative to the size of the document
            - upper bound on the dimensions relative to the size of the document

        :param x: x coordinate of the left segment of the upright bounding rectangle
        :param y: y coordinate of the upper segment of the upright bounding rectangle
        :param width: width of the upright bounding rectangle
        :param height: height of the upright bounding rectangle
        :param threshold: threshold to use while determining the pixel density
        :param rotated_rect: a tuple (center, (width, height), angle) determining the minimal are bounding rectangle
        around the object

        :returns: a tuple (flag, corners) where flag is true iff the tracked features satisfy the defined bounds and
        corners is the corner coordinates of the detected object
        """
        # if a threshold is not given we create one, this is the case in non-color types of methods, i.e. cascade
        # classifier and Hough Circle Transform
        if threshold is None:
            _, threshold = cv2.threshold(
                255 - self.grayscale, 140, 255, cv2.THRESH_BINARY)

        # if a more fitting rotated rectangle around an object is found we calculate the density of the rotated
        # rectangle to correctly reflect the pixel density of the object, as rotating is resource sensitive we do
        # this only if the angle of rotation is high enough
        if rotated_rect is not None:
            center, wh, angle = rotated_rect
            if 10 < angle < 80:
                rotated_object = self._rotate_image(
                    threshold, center, angle, int(wh[0]), int(wh[1]))
                density_in_bound = 0.064 < rotated_object.mean() / 255 <= 0.61
            # if the rectangle is only slightly rotated we calculate the density just based on the (upright) bounding
            # rectangle
            else:
                density_in_bound = 0.064 < threshold[y:y + height, x:x + width].mean() / 255 <= 0.5
        # if the rotated rectangle is not supplied we calculate the density just based on the (upright) bounding
        # rectangle
        else:
            density_in_bound = 0.064 < threshold[y:y + height, x:x + width].mean() / 255 <= 0.5

        # the width:height ratio is determined
        wh_ratio_in_bound = 0.3 < width / height <= 4

        # the relative dimensions should follow these rules
        dimensions_lower_bound = width > self.width / 20 and height > self.height / 50
        dimensions_upper_bound = width < self.width / 3 and height < self.height / 4

        # verify the features
        flag = density_in_bound and wh_ratio_in_bound and dimensions_lower_bound and dimensions_upper_bound

        # save the corner of the (upright) bounding rectangle of the object
        corners = [x, y, x + width, y + height]

        # return whether the area has stamp like features and the corners of the bounding box around the object
        return flag, corners

    def _remove_duplicates(self):
        """
        Some stamps may be detected several times using different methods, this function removes duplicates. The
        definition of a duplicate in this context is as follows: an area is considered a duplicate if there is a
        larger area such that their intersection is more than 1/4 of the smaller area.
        """

        # helper method to derive the area of a region
        def area(coordinates):
            return (coordinates[2] - coordinates[0]) * (coordinates[3] - coordinates[1])

        duplicates = []

        # we iterate over unordered pairs
        for pair in itertools.combinations(self.stamp_coordinates, 2):
            obj0, obj1 = pair
            if obj0 != obj1:
                # determine the areas and the smaller region
                area_obj0 = area(obj0)
                area_obj1 = area(obj1)
                smaller_obj = obj0 if area_obj0 < area_obj1 else obj1

                # if the regions do not overlap continue to next pair
                if obj0[2] <= obj1[0] or obj0[3] <= obj1[1] or obj1[2] <= obj0[0] or obj1[3] <= obj0[1]:
                    continue

                # calculate the intersection coordintes if there is an overlap
                intersection_coordinates = [max(obj0[0], obj1[0]), max(obj0[1], obj1[1]), min(obj0[2], obj1[2]),
                                            min(obj0[3], obj1[3])]
                # determine the are if the intersection
                area_intersection = area(intersection_coordinates)

                # if the are is big enough we mark the smaller region as a duplictae
                if area_intersection > 0.25 * min(area_obj0, area_obj1):
                    duplicates.append(smaller_obj)

        # remove all duplicates from the detected stamps
        self.stamp_coordinates = [
            x for x in self.stamp_coordinates if x not in duplicates]

    def _draw_stamp_rectangles(self):
        """
        Helper method to draw rectangle around stamp like objects found during the analysis.
        """
        for obj in self.stamp_coordinates:
            x1, y1, x2, y2 = obj
            cv2.rectangle(self.output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    def analyze(self):
        """
        Function utilizing all the methods used to find stamps in official documents.
        """
        # these are the blue and red (respectively) HSV color bounds used in `find_colored_objects_hsv`
        br_color_bounds = [[(90, 25, 100), (120, 255, 255)], [
            (150, 50, 0), (179, 255, 255)]]

        # first we look for circles
        self.find_circles()

        # then we find contours in the appropriate thresholds based on the blue and red HSV bounds
        for bounds in br_color_bounds:
            self.find_colored_objects_hsv(*bounds)

        # a second color analysis using the YCrCb color space
        self.find_colored_objects_ycbcr()

        # the cascade classifier is run to automatically detect areas resembling stamps
        self.cascade_classifier()

        # remove duplicates
        self._remove_duplicates()

        # draw bounding rectangles around stamps
        self._draw_stamp_rectangles()


def read_input():
    """
    Parser acting as a CLI for the analyzer. It takes as an input a path to a given document and performs an analysis
    for stamp detection, the results are returned as a list of 4-tuples (in JSON format) where each 4-tuple
    determines a rectangle around a detected stamp. The 4-tuple (x1, y1, x2, y2) determines a rectangle whose left
    segment has x coordinate x1, upper segment has y coordinates y1, right segment has x coordinate x2 and bottom
    segment has y coordinate y2.

    Usage: python3 main.py [-h] [-s] path

    Positional arguments:
        path        path to a document

    Optional arguments:
        -h, --help  show help message and exit
        -s, --show  show the detected feature within the document
    """
    # initialize the parser
    parser = argparse.ArgumentParser(
        description="Detect stamps in a given document")

    # add an argument for the path
    parser.add_argument("path", help="path to a document")

    # add an optional argument for displaying the stamps within the document
    parser.add_argument("-s", "--show", action="store_true",
                        help="show the detected feature within the document")

    # parse the input
    return parser.parse_args()


def show_image(img, scale=900, window_name='Detected stamps'):
    """
    Helper method to display an image with given scaling (the desired height of the image).

    The window is closed by pressing any key. 

    :param img: image to display
    :param scale: the height to which we want to scale the image
    :param window_name: the name of the window displaying the image
    """
    # display the image
    cv2.imshow(window_name, cv2.resize(
        img, (int(scale * (img.shape[1] / img.shape[0])), scale)))

    # wait for any key press
    cv2.waitKey(0)

    # close window
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # saved the parsed data
    args = read_input()

    # define the document from the input
    document = cv2.imread(args.path)

    # analyze the document
    analyzer = StampAnalyzer(document)

    # if there are detected objects
    if len(analyzer.stamp_coordinates) > 0:
        # print their coordinates
        print(analyzer.stamp_coordinates)

        # and display them if indicated by use
        if args.show:
            show_image(analyzer.output)
    # else notify the user about lack of detected objects
    else:
        print("No stamps found in this document")
