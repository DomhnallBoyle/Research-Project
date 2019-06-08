"""
    Filename: line/advanced_line.py
    Description: Contains functionality for creating and updating an instance of a polynomial line
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import collections
import cv2
import numpy as np

# local source imports
from line.base_line import BaseLine


class AdvancedLine(BaseLine):
    """Class that extends BaseLine that represents a polynomial line

    Based on:
    https://github.com/ndrplz/self-driving-car/tree/master/project_4_advanced_lane_finding

    Attributes:
        detected (Boolean): whether there is a line detected or not
        last_fit_pixel (List): list of polynomial co-efficients
        last_fit_meter (List): list of polynomial co-efficients with repect to the line in metres
        recent_fits_pixel (Deque): collection of polynomial co-efficients
        recent_fits_meter (Deque): collection of lists of polynomial co-efficients with respect to the line in metres
        radius_of_curvature (Float): radius of the curvature of the line
        all_x (List): store the x pixel coordinates of the detected line
        all_y (List): store the corresponding y pixel coordinates of the detected line
    """

    def __init__(self, buffer_len=10):
        """Instantiating an instance of AdvancedLine

        Calls the __init__ of the base class BaseLine

        Args:
            buffer_len (Integer): the max size of the deque to hold previous line co-efficients
        """
        super().__init__()
        # flag to mark if the line was detected the last iteration
        self.detected = False

        # polynomial coefficients fitted on the last iteration
        self.last_fit_pixel = None
        self.last_fit_meter = None

        # list of polynomial coefficients of the last N iterations
        self.recent_fits_pixel = collections.deque(maxlen=buffer_len)
        self.recent_fits_meter = collections.deque(maxlen=2 * buffer_len)

        self.radius_of_curvature = None

        # store all pixels coords (x, y) of line detected
        self.all_x = None
        self.all_y = None

    def midpoint(self):
        """Implemented abstract method from the base class

        Did not have time to use the advanced line with the trivial steering controller
        """
        pass

    def draw(self, image, color=(255, 0, 0), line_width=50, average=False):
        """Draw the line on a color mask image.

        Args:
            image (List): 3D list representing an RGB image
            color (Tuple): RGB colour
            line_width (:
            average:

        Returns:
            (List): RGB image with the lane drawn onto it
        """
        # get the height, width and channels of the image
        h, w, c = image.shape

        # return image height evenly spaced numbers over a specified interval
        plot_y = np.linspace(0, h - 1, h)

        # get the average of the co-efficients in the deque to draw
        coeffs = self.average_fit if average else self.last_fit_pixel

        # calculate the centre, left and right sides of the line
        line_center = coeffs[0] * plot_y ** 2 + coeffs[1] * plot_y + coeffs[2]
        line_left_side = line_center - line_width // 2
        line_right_side = line_center + line_width // 2

        # Some magic here to recast the x and y points into usable format for cv2.fillPoly()
        # all the left points of the line
        pts_left = np.array(list(zip(line_left_side, plot_y)))

        # all the right points of the line - flipped
        pts_right = np.array(np.flipud(list(zip(line_right_side, plot_y))))

        # vstack the left and right points
        pts = np.vstack([pts_left, pts_right])

        # draw the lane onto the warped blank image
        return cv2.fillPoly(image, [np.int32(pts)], color)

    def update_line(self, new_fit_pixel, new_fit_meter, detected, clear_buffer=False):
        """Update Line with new fitted coefficients.

        Args:
            new_fit_pixel: new polynomial coefficients (pixel)
            new_fit_meter: new polynomial coefficients (meter)
            detected: if the Line was detected or inferred
            clear_buffer: if True, reset state

        Returns:
            None
        """
        # update if the line is detected or not
        self.detected = detected

        # reset the coefficients if clear_buffer == True
        if clear_buffer:
            self.recent_fits_pixel = []
            self.recent_fits_meter = []

        # set the last co-efficients to the new ones
        self.last_fit_pixel = new_fit_pixel
        self.last_fit_meter = new_fit_meter

        # append the co-efficients to the deque
        self.recent_fits_pixel.append(self.last_fit_pixel)
        self.recent_fits_meter.append(self.last_fit_meter)

    @property
    def average_fit(self):
        """Calculate the average of polynomial coefficients of the last N iterations

        Returns:
            (Float): average of the co-efficients
        """
        return np.mean(self.recent_fits_pixel, axis=0)

    @property
    def curvature(self):
        """Calculate the radius of curvature of the line (averaged)

        Returns:
            (Float): radius of the line curvature
        """
        y_eval = 0
        coeffs = self.average_fit

        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])

    @property
    def curvature_meter(self):
        """Calculate the radius of curvature of the line (averaged)

        Returns:
            (Float): radius of the line curvature in metres
        """
        y_eval = 0
        coeffs = np.mean(self.recent_fits_meter, axis=0)

        return ((1 + (2 * coeffs[0] * y_eval + coeffs[1]) ** 2) ** 1.5) / np.absolute(2 * coeffs[0])
