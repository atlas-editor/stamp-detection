"""
Provides utility functions to prepare files for training a cascade classifier using `opencv_traincascade`,
see https://docs.opencv.org/3.4/dc/d88/tutorial_traincascade.html.
"""
import os
import random
from PIL import Image


def crop_non_stamp_images(src, dest, N=100):
    """
    Given documents which DO NOT contain stamps, crop random rectangles from them and save them to create negative
    training data for a cascade classifier.

    Redundant method as the `opencv_traincascade` utility does this automatically.

    :param src: the directory containing the negative images
    :param dest: the directory where cropped images will be saved
    :param N: the number of cropped images from one single document
    """

    def crop_random_rectangle(img, w_min=300, w_max=700, h_min=150, h_max=300):
        """
        Crops a random rectangle from a given image.
        
        For our purposes (stamp detection) the random rectangle has width in the interval [300,700] and height in the
        interval [150,300].

        :param img: the image to be cropped
        :param w_min: the minimal width of the rectangle
        :param w_max: the maximal width of the rectangle
        :param h_min: the minimal height of the rectangle
        :param h_max: the maximal height of the rectangle
        :returns: the cropped image `img`
        """
        image = Image.open(img)
        image_width, image_height = image.size

        # first we choose the (x,y) coordinates of the upper left corner
        x = random.randint(0, image_width - w_max)
        y = random.randint(0, image_height - h_max)

        # we randomize the width and the height of the rectangle
        crop_width = random.randint(w_min, w_max)
        crop_height = random.randint(h_min, h_max)

        return image.crop((x, y, x + crop_width, y + crop_height))

    for filename in os.listdir(src):
        img = src + filename
        for j in range(N):
            cropped = crop_random_rectangle(img)
            cropped.save(f"{dest}crop-{j}-{filename}")

        print(f"image {filename} cropped and saved")


def crop_stamp(map, pixel):
    """
    Given two (uncropped) images (both coming from scans of documents), map and pixel (both grayscale), where pixel
    contains the stamp and map is shaded black on a rectangle (may be rotated) covering the stamp (if the two images
    were superimposed) this methods finds the region of the image where the stamp is present and crops the image to
    contain only the stamp region.

    :param map: the image which shows the part of the scan where the stamp is present
    :param pixel: the image which shows only the stamp on the scan
    :returns: the cropped stamp
    """

    image = Image.open(map)
    width, height = image.size

    # u, r, b, l will denote the upper, right, bottom and left corners of the stamp within the image
    # u is the y coordinate of the upper corner (possibly the entire upper segment of the rectangle if not rotated)
    # r is the x coordinate of the right corner ...
    # etc.
    u, r, b, l = -1, -1, -1, -1

    # we loop over the entire image pixel by pixel and find the extremes of the shaded region
    for y in range(height):
        for x in range(width):
            color = image.getpixel((x, y))
            print(color)
            if color == 0:
                if u == -1:
                    u, r, b, l = y, x, y, x
                r = max(r, x)
                b = max(b, y)
                l = min(l, x)

    original = Image.open(pixel)
    if u == -1:
        return None
    return original.crop((l, u, r, b))


def crop_saved_stamps(map_dir, pixel_dir, info_dir, cropped_dir):
    """
    Given documents containing stamps and further data (see the `crop_stamp` method) crop the stamps from the scans
    and save them. These are to be used as positive training data for a cascade classifier.

    :param map_dir: directory containing the maps, see map in `crop_stamp`
    :param pixel_dir: directory containing the pixels, see pixel in `crop_stamp`
    :param info_dir: directory containing info about stamps in a given document
    :param cropped_dir: directory where the cropped stamps will be saved
    """
    for i in range(400, 401):
        map = f"./{map_dir}/stampDS-{str(i).rjust(5, '0')}-gt.png"
        pixel = f"./{pixel_dir}/stampDS-{str(i).rjust(5, '0')}-px.png"

        with open(f"./{info_dir}/stampDS-{str(i).rjust(5, '0')}.txt", 'r') as info:
            info.readline()  # header; the format is: "signature    textOverlap    numStamps    bwStamp1"
            no_of_stamps = int(info.readline().strip().split()[
                                   2])  # the number of stamps
            # we only care about documents with exactly 1 stamp
            if no_of_stamps == 0:
                print(f"image no. {i} has no stamp")
                continue
            elif no_of_stamps > 1:
                print(f"image no. {i} has more than one stamp")
                continue

        cropped = crop_stamp(map, pixel)
        if cropped is None:
            print(
                f"image no. {i} has no stamp but the info file indicated otherwise")
            continue
        cropped.save(
            f"./{cropped_dir}/stampDS-{str(i).rjust(5, '0')}-cr.png")

        print(f"image no. {i} saved")

    print('\nall stamps cropped')


def create_negative_list(classifier_dir):
    """
    Create a document containing the negative images for the `opencv_traincascade` utility.

    :param classifier_dir: directory containing the classifier
    """
    with open(f"./{classifier_dir}/bg.txt", 'w') as file:
        for img in os.listdir(f"./{classifier_dir}/bg/"):
            file.write(f"./bg/{img}\n")


def create_positive_list(classifier_dir):
    """
    Create a document containing the positive images (those which contain the desired object) for the
    `opencv_traincascade` utility.

    :param classifier_dir: directory containing the classifier
    """
    with open(f"./{classifier_dir}/info.dat", 'w') as file:
        for img in os.listdir("./{classifier_dir}/p/"):
            image = Image.open(f"./{classifier_dir}/p/{img}")
            file.write(f"./p/{img} 1 0 0 {image.size[0]} {image.size[1]}\n")


if __name__ == "__main__":
    pass
