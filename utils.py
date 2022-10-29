import itertools
from typing import List
import cv2


def show_image(img: cv2.Mat, scale: int = 900, window_name: str = 'Detected stamps') -> None:
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


def rotate_image(image: cv2.Mat, center: List[int], theta: float, width: int, height: int) -> cv2.Mat:
    """
        Rotates OpenCV image around center with angle theta (in deg) then crops the image according to width and height.

        Source: https://stackoverflow.com/a/11627903.

        :param image: image to rotate
        :param center: center of rotation
        :param theta: angle of rotation
        :param width: width of the region we are cropping
        :param height: height of the region we are cropping

        :returns: the rotated image
        """

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

def create_binary_image(image: cv2.Mat, threshold_lower_bound: int = 140, threshold_upper_bound: int = 255) -> cv2.Mat:
    """
    Given an image create a binary version using thresholding with given values.

    :param image: the image to binarize
    :param threshold_lower_bound: thresholding lower bound
    :param threshold_upper_bound: thresholding upper bound

    :returns: binary version of the image
    """
    _, threshold = cv2.threshold(image, threshold_lower_bound, threshold_upper_bound, cv2.THRESH_BINARY)

    return threshold


def remove_duplicates(stamp_coordinates: List[List[int]]) -> None:
    """
        Some stamps may be detected several times using different methods, this function removes duplicates. The
        definition of a duplicate in this context is as follows: an area is considered a duplicate if there is a
        larger area such that their intersection is more than 1/4 of the smaller area.

        :param stamp_coordinates: list of 4-tuples of coordinates of stamps
        """

    # helper method to derive the area of a region
    def area(coordinates: List[List[int]]) -> int:
        return (coordinates[2] - coordinates[0]) * (coordinates[3] - coordinates[1])

    duplicates = []

    # we iterate over unordered pairs
    for pair in itertools.combinations(stamp_coordinates, 2):
        obj0, obj1 = pair
        if obj0 != obj1:
            # determine the areas and the smaller region
            area_obj0 = area(obj0)
            area_obj1 = area(obj1)
            smaller_obj = obj0 if area_obj0 < area_obj1 else obj1

            # if the regions do not overlap continue to next pair
            if obj0[2] <= obj1[0] or obj0[3] <= obj1[1] or obj1[2] <= obj0[0] or obj1[3] <= obj0[1]:
                continue

            # calculate the intersection coordinates if there is an overlap
            intersection_coordinates = [max(obj0[0], obj1[0]), max(obj0[1], obj1[1]), min(obj0[2], obj1[2]),
                                        min(obj0[3], obj1[3])]
            # determine the area if the intersection
            area_intersection = area(intersection_coordinates)

            # if the are is big enough we mark the smaller region as a duplicate
            if area_intersection > 0.25 * min(area_obj0, area_obj1):
                duplicates.append(smaller_obj)

    # remove all duplicates from the detected stamps
    stamp_coordinates = [x for x in stamp_coordinates if x not in duplicates]


def draw_stamp_rectangles(stamp_coordinates: List[List[int]], img: cv2.Mat) -> cv2.Mat:
    """
        Helper method to draw a rectangle around stamp like objects found during the analysis.

        :param stamp_coordinates: list of 4-tuples of coordinates of stamps
        :param img: the image where the stamps were detected and the rectangles will be drawn

        :returns: a copy of the input `img` with rectangles drawn around stamps as specified in `stamp_coordinates`
        """
    output = img.copy()
    for obj in stamp_coordinates:
        x1, y1, x2, y2 = obj
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)

    return output
