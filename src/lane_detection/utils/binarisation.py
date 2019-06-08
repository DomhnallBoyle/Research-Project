"""
    Filename: lane_detection/utils/binarisation.py
    Description: Contains functionality for thresholding images, applying masks, resulting in binary images
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


def threshold_HSV_image(image, min_value, max_value, verbose=False):
    """Apply thresholding to an image converted to the HSV colour space using min and max intervals

    Find all pixel values > minimum value AND < maximum value

    Args:
        image (List): 3D list representing an RGB image
        min_value (Integer): minimum threshold value
        max_value (Integer): maximum threshold value
        verbose (Boolean): for debugging purposes

    Returns:
        (List): 2D list representing a grayscale image
    """
    # convert to HSV
    HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # get minimum and maximum thresholds from the 2nd channel (saturation/amount of gray)
    min_th_ok = np.all(HSV > min_value, axis=2)
    max_th_ok = np.all(HSV < max_value, axis=2)

    # logical AND both results
    result = np.logical_and(min_th_ok, max_th_ok)

    # display results if in debug mode
    if verbose:
        images = [HSV, min_th_ok, max_th_ok, result]
        titles = ['HSV', 'Min Threshold', 'Max Threshold', 'Logical AND']
        cv2_show_figured_images(images=images, titles=titles)

    return result


def threshold_equalised_image(image, threshold, verbose=False):
    """Apply histogram equalisation to the image and threshold it

    Args:
        image (List): 3D list representing an RGB image
        threshold (Integer): the min value threshold from the image
        verbose (Boolean): for debugging purposes

    Returns:
        (List): 3D list representing an RGB image
    """
    # convert image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply histogram equalisation to improve image contrast
    equalised_image = cv2.equalizeHist(grayscale)

    # threshold the grayscale image looking for pixels between threshold and 255
    # doing binary thresholding so white pixels because of the max_value being pure white
    _, thresholded_image = cv2.threshold(equalised_image, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)

    # display results if in debug mode
    if verbose:
        images = [image, grayscale, equalised_image, thresholded_image]
        titles = ['Original', 'Grayscale', 'Equalised', 'Thresholded']
        cv2_show_figured_images(images=images, titles=titles)

    return thresholded_image


def threshold_sobel_image(image, threshold, kernel_size, verbose=False):
    """Find the edges using Sobel edge detection and apply binary thresholding to get the clearest edges

    Args:
        image (List): 3D list representing an RGB image
        threshold (Integer): for binary thresholding - minimum value used to threshold the best sobel edges
        kernel_size (Integer): size of the convolutional kernel mask
        verbose (Boolean): for debugging purposes

    Returns:
        (List): 3D list representing an RGB image
    """
    # convert image to grayscale
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # apply sobel masks on image - get horizontal and vertical edges
    sobel_x = cv2.Sobel(grayscale, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sobel_y = cv2.Sobel(grayscale, cv2.CV_64F, 0, 1, ksize=kernel_size)

    # combine horizontal and vertical edges by magnitude
    sobel_mag = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    sobel_mag = np.uint8(sobel_mag / np.max(sobel_mag) * 255)

    # apply binary thresholding to the sobel edges
    _, sobel_image = cv2.threshold(sobel_mag, thresh=threshold, maxval=255, type=cv2.THRESH_BINARY)

    # display results if in debug mode
    if verbose:
        images = [image, grayscale, sobel_x, sobel_y, sobel_mag, sobel_image]
        titles = ['Original', 'Grayscale', 'Sobel X', 'Sobel Y', 'Sobel Mag', 'Sobel Threshold']
        cv2_show_figured_images(images=images, titles=titles)

    return sobel_image


def apply_morphology(binary_image, kernel_size, morphology_type, verbose=False):
    """Apply morphology to the binary image for open/closing the grouped non-zero pixels

    Results in cleaner images

    Args:
        binary_image (List): 2D list representing a grayscale image
        kernel_size (Integer):
        morphology_type (Integer): type of morphology to apply (open/closed)
        verbose (Boolean): for debugging purposes

    Returns:
        (List): 2D list representing a grayscale image
    """
    # create a kernel of 1's using the specified kernel size
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # apply morphology using the type and kernel
    morphology_results = cv2.morphologyEx(binary_image.astype(np.uint8), morphology_type, kernel)

    # display results if in debug mode
    if verbose:
        images = [morphology_results]
        titles = ['Morphology Results']
        cv2_show_figured_images(images=images, titles=titles)

    return morphology_results
