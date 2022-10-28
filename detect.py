import argparse

import cv2
from stamp_analyzer import StampAnalyzer

from utils import draw_stamp_rectangles, show_image


def read_input():
    """
    Parser acting as a CLI for the analyzer. It takes as an input a path to a given document and performs an analysis
    for stamp detection.

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


def main():
    """
    This defines the high-level processes. We read the input via the parser, define the document and the analyzer and
    print the results.
    """
    # saved the parsed data
    args = read_input()

    # define the document from the input
    document = cv2.imread(args.path)

    # analyze the document
    analyzer = StampAnalyzer(document)

    stamp_coordinates = analyzer.analyze()

    # if there are detected objects
    if len(stamp_coordinates) > 0:
        # print their coordinates
        print(stamp_coordinates)

        # and display them if indicated by user
        if args.show:
            output = draw_stamp_rectangles(stamp_coordinates, document)
            show_image(output)
    # else notify the user about lack of detected objects
    else:
        print("No stamps found in this document")


if __name__ == '__main__':
    main()
