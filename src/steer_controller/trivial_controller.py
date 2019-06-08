"""
    Filename: steer_controller/trivial_controller.py
    Description: Trivial controller for predicting the steering angle given 2 detected lanes
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import cv2
import math
import numpy as np

# local source imports
from steering_controller import SteeringController


class TrivialController(SteeringController):
    """Concrete class that extends the SteeringController base class

    Overrides the get_steering_angle() method. Based on the trivial controller found in:
    https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7091469&tag=1

    Attributes:
        None
    """

    def __init__(self):
        """Instantiating an instance of TrivialController

        Calls the __init__ of the base class SteeringController
        """
        super().__init__()

    def get_steering_angle(self, image, left_lane, right_lane):
        """Overridden method from the base class

        Calculates the steering angle based on the centre of the image and centre of the lanes

        Args:
            image (List): 3D list representing an RGB image
            left_lane (BaseLine): Detected left-lane object
            right_lane (BaseLine): Detected right-lane object

        Returns:
            (Tuple): containing the image and the predicted steering angle
        """
        # get the midpoint of both left and right detected lanes
        left_lane_midpoint = left_lane.midpoint()
        right_lane_midpoint = right_lane.midpoint()

        # draw the midpoints on the image as circles
        # draw a line between both midpoints
        cv2.circle(image, left_lane_midpoint, 5, (0, 0, 255), -1)
        cv2.circle(image, right_lane_midpoint, 5, (0, 0, 255), -1)
        cv2.line(image, left_lane_midpoint, right_lane_midpoint, (0, 0, 255), 2)

        # get the midpoint of this line - centre of the lane
        line_midpoint = self.midpoint(left_lane_midpoint, right_lane_midpoint)

        # find the bottom-centre of the image
        image_bottom_midpoint = (int(1280 / 2), 720)

        # draw a circle at this point
        # draw a line from the bottom-centre to the centre of the lane
        cv2.circle(image, image_bottom_midpoint, 5, (0, 0, 255), -1)
        cv2.line(image, image_bottom_midpoint, line_midpoint, (255, 0, 0, 2))

        # get the bottom of the centre of the line
        line_bottom_midpoint = (line_midpoint[0], 720)
        cv2.circle(image, line_bottom_midpoint, 5, (0, 0, 255), -1)
        cv2.line(image, line_midpoint, line_bottom_midpoint, (255, 0, 0, 2))

        # get the euclidean distance of hypotenuse and opposite lengths
        hyp = np.linalg.norm(np.asarray(image_bottom_midpoint) - np.asarray(line_midpoint))
        opp = np.linalg.norm(np.asarray(image_bottom_midpoint) - np.asarray(line_bottom_midpoint))

        # inverse sine and get the angle in radians
        steering_angle = math.asin(opp / hyp)

        # reverse the angle depending on the steering angle direction
        if image_bottom_midpoint[0] > line_bottom_midpoint[0]:
            steering_angle = -steering_angle

        return image, steering_angle

    def midpoint(self, p1, p2):
        """Get the midpoint between two points

        Args:
            p1 (Tuple): (x, y) point 1
            p2 (Tuple): (x, y) point 2

        Returns:
            (Tuple): containing the (x,y) coordinates of the midpoint between 2 points
        """
        return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
