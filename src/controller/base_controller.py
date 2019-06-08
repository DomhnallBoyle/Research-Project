"""
    Filename: controller/base_controller.py
    Description: Contains functionality for the base controller base class
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
from abc import ABC, abstractmethod


class BaseController(ABC):
    """Abstract base class containing generic attributes and abstract functions to be overridden by derived controllers

    Attributes:
        debug (Boolean): for debugging purposes of the controller
    """

    def __init__(self, debug):
        """Abstract base class - cannot instantiate instance of abstract class

        Args:
            debug (Boolean): for debugging purposes of the controller
        """
        self.debug = debug

    @abstractmethod
    def get_steering_angle(self, *args):
        """Abstract method to be implemented by each derived controller to get the steering angle given a list of
        arguments

        Args:
            *args (Tuple): contains arguments specific to the method implemented by the derived controller

        Raises:
            NotImplementedError: if the method has not been overridden by the derived class
        """
        raise NotImplementedError
