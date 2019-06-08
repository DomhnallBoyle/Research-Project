"""
    Filename: steer_controller/edt_controller.py
    Description: Controller for predicting the steering angle given 2 detected lanes using Euclidean Distance Transform
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# local source imports
from steering_controller import SteeringController


class EDTController(SteeringController):
    """Class for controlling the vehicles using detected lanes and a technique called the Euclidean Distance Transform

    Did not have time to complete this functionality

    Based on the approach from:
    # https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7091469&tag=1

    Attributes:
        None
    """

    def __init__(self):
        """Instantiating an instance of EDTController

        Calls the __init__ of the base class SteeringController
        """
        super().__init__()

    def get_steering_angle(self, image, left_lane, right_lane):
        """Overridden method from the base class

        Calculates and returns the steering angle based on the detected lanes

        Args:
            image (List): 3D list representing an RGB image
            left_lane (BaseLine): Detected left-lane object
            right_lane (BaseLine): Detected right-lane object

        Returns:
            (Tuple): containing the image and the predicted steering angle
        """
        pass
