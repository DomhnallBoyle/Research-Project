"""
    Filename: lane_detection/utils/general.py
    Description: Contains general functions for lane detection
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import cv2
import numpy as np
import json

# global constants
WIDTH, HEIGHT = (1280, 720)


def coords(s):
    """Convert the command line argument ROI coords to an array of tuples

    Args:
        s (String): in the form of a list from the command line

    Returns:
        (List): containing tuples of coordinates
    """
    # load the string to as JSON
    horizon = json.loads(s)

    # append each coordinate to the list as a tuple
    coords = []
    for coord in horizon:
        coords.append(tuple(coord))

    # convert to numpy array
    vertices = np.array([coords], dtype=np.int32)

    return vertices


def resize(image):
    """Function for resizing an image

    Args:
        image (List): representing an image (2D or 3D)

    Returns:
        (List): resized image
    """
    # gets the current width and height of the image
    height, width = image.shape[:2]

    # don't resize if already the same width and height
    if height != HEIGHT or width != WIDTH:
        return cv2.resize(image, (WIDTH, HEIGHT))

    return image
