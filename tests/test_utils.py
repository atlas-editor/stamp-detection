import numpy as np
from utils import create_binary_image, remove_duplicates, rotate_image


def test_rotate_image() -> None:
    # define a 3x3 image with all zeroes except in the middle of the right segment
    image = np.zeros([3, 3, 3], dtype=np.uint8)
    image[1][2] = [1, 1, 1]

    # define a 3x3 image with all zeroes except in the middle of the upper segment; also the counter clockwise rotation
    # of the previous `image` by 90 degrees
    image_90 = np.zeros([3, 3, 3], dtype=np.uint8)
    image_90[0][1] = [1, 1, 1]

    center = [1, 1]
    angle = 90.0
    width = 9
    height = 9

    # rotate image 90 degrees counter clockwise
    rotated_image = rotate_image(image, center, angle, width, height)

    assert np.equal(image_90, rotated_image).all()


def test_create_binary_image() -> None:
    # we define a random grayscale image
    random_square = np.random.randint(1, 255, [10, 10], dtype=np.uint8)
    stamp = np.dstack(3 * [random_square])
    binary_stamp = create_binary_image(stamp)

    # the binary image should only have 2 values
    assert np.isin(binary_stamp, [0, 255]).all()


def test_remove_duplicates() -> None:
    # several overlapping (>= 25% of area of smaller region) areas are defined w/ the square [10, 10, 20, 20] having the
    # largest are among all and
    coordinates = [[10, 10, 20, 20], [14, 14, 20, 20], [11, 11, 20, 20], [10, 10, 15, 15]]

    coordinates = remove_duplicates(coordinates)

    assert len(coordinates) == 1
