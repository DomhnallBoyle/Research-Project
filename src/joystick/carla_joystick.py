"""
    Filename: joystick/carla_joystick.py
    Description: Contains functionality for controlling the CARLA simulator with the PS3 controller
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import math

# local source imports
from base_joystick import BaseJoystick


class CarlaJoystick(BaseJoystick):
    """Class that extends the BaseJoystick containing functionality to control the CARLA simulator through the PS3
    controller

    Implements abstract methods from the base class
    """

    def __init__(self):
        """Instantiate an instance of the CarlaJoystick to control the simulator

        Calls the __init__ of the BaseJoystick class
        """
        super().__init__()

    def _get_controls(self):
        """Implemented abstract method from the base class

        Takes the joystick inputs and creates output commands that are understandable for the CARLA simulator

        Based on:
        https://gist.github.com/sandman/366e45d3a836da1a2a90fe9eeccd689a

        Returns:
            (Tuple): suitable steering angle, brake and throttle controls for the CARLA simulator
        """
        steer = 0.55 * math.tan(1.1 * self.joystick_inputs[0])
        brake = (((self.joystick_inputs[2] - (-1)) * (1.0 - 0)) / (1.0 - (-1.0))) + 0
        throttle = (((self.joystick_inputs[5] - (-1)) * (1.0 - 0)) / (1.0 - (-1.0))) + 0

        return steer, brake, throttle
