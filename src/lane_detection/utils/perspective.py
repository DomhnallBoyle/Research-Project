"""
    Filename: lane_detection/utils/perspective.py
    Description: Contains general functions for changing the perspective of the image
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import cv2
import numpy as np

# local source imports
from draw import *


def get_roi(image, vertices, verbose=False):
    """Get the Region of Interest of an image based on coords

    Args:
        image (List): 2D list representing an grayscale image
        vertices (List): contains tuples of (x, y) coords that contain the ROI
        verbose (Boolean): for debugging purposes

    Returns:
        (List): 2D list representing the extracted ROI of the grayscale image
    """
    # create blank mask with the image's size
    mask = np.zeros_like(image)

    # filling pixels inside the polygon defined by "vertices" with the fill color (255) = ignore mask colour
    cv2.fillPoly(mask, vertices, 255)

    # apply bitwise and between image and mask
    masked = cv2.bitwise_and(image, mask)

    # display results if in debug mode
    if verbose:
        images = [image, masked]
        titles = ['Before', 'After']
        cv2_show_figured_images(images=images, titles=titles)

    return masked


def get_birds_eye(image, coords, verbose=False):
    """Get the birds eye view of the image

    Transforms the image to a top down view using the coords

    Args:
        image (List): 2D list representing a grayscale image
        coords (List): ROI coords that will be mapped to a top down view
        verbose (Boolean): for debugging purposes

    Returns:
        (Tuple): the warped image and inverse transform to warp back
    """
    # get the image dimensions
    h, w = image.shape[:2]

    # the idea is to map the source coordinates of the ROI to the destination coordinates of the image dimensions
    src = []
    for coord in coords[0]:
        src.append(coord)
    src = np.float32([src])

    # the destination coords to map the source coords to
    dst = np.float32([[0, h],  # bl
                      [0, 0],  # tl
                      [w, 0],  # tr
                      [w, h]])  # br

    # get the transform and inverse transform co-efficients (3x3 matrix)
    transform = cv2.getPerspectiveTransform(src, dst)
    transform_inversed = cv2.getPerspectiveTransform(dst, src)

    # apply the perspective transform to the image keeping the same width and height
    warped = cv2.warpPerspective(image, transform, (w, h), flags=cv2.INTER_LINEAR)

    # display results if in debug mode
    if verbose:
        images = [image, warped]
        titles = ['Before', 'After']
        cv2_show_figured_images(images=images, titles=titles)

    return warped, transform_inversed
