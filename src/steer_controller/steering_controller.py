"""
    Filename: steer_controller/steering_controller.py
    Description: Contains functionality to be inherited by concrete steering controllers
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
from abc import ABC, abstractmethod


class SteeringController(ABC):
    """Abstract base class that contains functionality to be overridden for controlling the vehicle based on detected
    lanes.

    Attributes:
        None
    """

    def __init__(self):
        """Cannot instantiate an instance of an abstract base class

        Args:
            None
        """
        pass

    @abstractmethod
    def get_steering_angle(self, image, left_lane, right_lane):
        """Abstract method to be implemented by base classes

        Returns a steering angle based on detected lanes

        Args:
            image (List): 3D list representing an RGB image
            left_lane (BaseLine): Detected left-lane object
            right_lane (BaseLine): Detected right-lane object

        Raises:
            NotImplementedError if the function has not been overridden by the subclass
        """
        raise NotImplementedError
