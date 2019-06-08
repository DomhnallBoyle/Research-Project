"""
    Filename: joystick/wheelchair_joystick.py
    Description: Contains functionality for controlling the joystick using the PS3 controller
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import pygame

# local source imports
from base_joystick import BaseJoystick


class WheelchairJoystick(BaseJoystick):
    """Class that extends the BaseJoystick containing functionality to control the Wheelchair through the PS3 controller

    Implements abstract methods from the base class
    """

    def __init__(self):
        """Instantiate an instance of the WheelchairJoystick to control the Wheelchair

        Calls the __init__ of the BaseJoystick class
        """
        super().__init__()

    def _get_controls(self, **kwargs):
        """Implemented abstract method from the base class

        Takes the joystick inputs and creates output commands that are understandable for the wheelchair

        Args:
            **kwargs (Dictionary): Contains key-word arguments specific to this joystick e.g. max-speed

        Returns:
            (Tuple): contains the steering angle and speed for the API
        """
        # check if there is any direction given on the joystick
        if not all(joystick_input == 0 for joystick_input in self.joystick_inputs[:2]):
            # create a vector from the left joystick inputs from the controller
            vector = pygame.math.Vector2(self.joystick_inputs[0], self.joystick_inputs[1])

            # extract the angle from the joystick
            radius, angle = vector.as_polar()

            # angle manipulation to get it angle between -90 and 90
            angle += 90
            if 90 < angle < 180:
                angle = 90
            elif 180 < angle <= 270:
                angle = -90

            # wheelchair defect - make sure angle not greater than 70 degrees either way
            if angle < 0:
                # normalise the negative angle between -70 and 0
                angle = self.normalise(angle, -90, 0, -70, 0)
            elif angle > 0:
                # normalise the positive angle between 0 and 70
                angle = self.normalise(angle, 0, 90, 0, 70)

            # wheelchair controls flipped. Left = positive, right = negative
            angle = -angle
        else:
            # just go straight
            angle = 0

        # speed manipulation
        # if the right trigger on the joystick has been pressed
        if self.joystick_inputs[5] not in [-1, 0]:
            # some manipulation on the throttle - speed between 0 and 1
            throttle = (self.joystick_inputs[5] + 1) / 2.0
            # normalise the speed between 0 and max_speed
            speed = self.normalise(throttle, 0, 1, 0, kwargs['max_speed'])
        elif self.joystick_inputs[2] not in [-1, 0]:
            # left trigger pressed on the joystick - reverse speed
            reverse = (self.joystick_inputs[2] + 1) / 2.0
            # normalise the throttle from 0 and 1 to -30 and 0
            speed = self.normalise(reverse, 0, 1, -30, 0)
            # want the reverse speed to get faster when we press down on the trigger even more
            speed = -30 + abs(speed)
        else:
            # no press on either left or right triggers
            speed = 0

        # convert to integers to make them understandable for the API
        angle = int(angle)
        speed = int(speed)

        return int(angle), int(speed)
