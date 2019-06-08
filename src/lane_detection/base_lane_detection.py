"""
    Filename: lane_detection/base_lane_detection.py
    Description: Abstract base class for the lane detectors
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
from abc import ABC, abstractmethod
import os
import sys

# for local source imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# local source imports
from lane_detection.utils.calibration import calibrate_camera


class BaseLaneDetection(ABC):
    """Abstract base class for the lane detection algorithms

    Attributes:
        debug (Boolean): for debugging purposes
        camera_calibration (Dictionary): contains calibration co-efficients for the images
        mtx (List): 3x3 floating-point camera matrix
        dist (List): list of distortion co-efficients
    """

    def __init__(self, debug):
        """Abstract base class - cannot instantiate

        Args:
            debug (Boolean): flag used for debugging purposes
        """
        self.debug = debug
        self.camera_calibration = calibrate_camera()
        self.mtx, self.dist = (self.camera_calibration['mtx'], self.camera_calibration['dist'])

    @abstractmethod
    def detect_lanes(self, image, roi_coords):
        """Abstract method to be implemented by the derived lane detectors

        To return the detected lane objects from the image

        Args:
            image (List): 3D list representing an RGB image
            roi_coords (List): contains tuples of coordinates used to extract the region-of-interest from the image

        Raises:
            NotImplementedError: if the method is not overridden by the base class
        """
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, image):
        """Abstract method to be implemented by the derived lane detectors

        Apply pre-processing techniques before extracting the lanes

        Args:
            image (List): 3D list representing an RGB image

        Raises:
            NotImplementedError: if the method is not overridden by the base class
        """
        raise NotImplementedError

    @abstractmethod
    def get_lanes(self, image):
        """Abstract method to be implemented by the derived lane detectors

        Detects the lanes from the pre-processed image and returns their object representations

        Args:
            image (List): 3D list representing an RGB image

        Raises:
            NotImplementedError: if the method is not overriden by the base class
        """
        raise NotImplementedError
