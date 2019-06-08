"""
    Filename: steer_controller/pid_controller.py
    Description: Controller for predicting the steering angle given 2 detected lanes using PID
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import numpy as np

# local source imports
from steering_controller import SteeringController

# global constants
YM_PER_PIX = 30 / 720   # meters per pixel in y dimension
XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension


class PIDController(SteeringController):
    """Class for controlling the vehicles using detected lanes and a PID controller

    The PID controller tries to reduce the offset from the centre of the lane (cross-track error)

    Based off the work from:
    https://github.com/ndrplz/self-driving-car/blob/master/project_9_PID_control/src/PID.cpp

    Attributes:
        k_p (Float): proportional co-efficient param to be optimised
        k_i (Float): integral co-efficient param to be optimised
        k_d (Float): derivative co-efficient param to be optimised
        error_proportional (Float): error proportional to the cross-track error
        error_integral (Float): helps to rectify steering drift if knocked off course
        error_derivative (Float): helps to not overshoot the x-axis
    """

    def __init__(self, k_p=0.1, k_i=0.0001, k_d=1.0):
        """Instantiating an instance of PIDController

        Calls the __init__ of the base class SteeringController

        Args:
            k_p: proportional co-efficient param
            k_i: integral co-efficient param
            k_d: derivative co-efficient param
        """
        super().__init__()

        # coefficients to be optimised
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.error_proportional = 0.0
        self.error_integral = 0.0
        self.error_derivative = 0.0

    def update_error(self, cte):
        """Updates the errors in relation to the offset of the center of the lane (cross-track-error)

        Args:
            cte (Float): Offset from the centre of the lane (cross-track-error)

        Returns:
            None
        """
        # added bias to rectify steering drift
        self.error_integral += cte

        # resistance for smoother transition to centre of lane
        self.error_derivative = cte - self.error_proportional

        # proportional to the cross-track-error
        self.error_proportional = cte

    def predict_steering(self):
        """Predict the steering angle using the updated errors

        Returns:
            (Float): steering angle to reduce the offset from centre
        """
        # multiply the updated errors by their co-efficients
        p = self.k_p * self.error_proportional
        i = self.k_i * self.error_integral
        d = self.k_d * self.error_derivative

        # add them together
        return -(p + i + d)

    def get_steering_angle(self, image, left_lane, right_lane):
        """Overridden method from the base class

        Calculates the steering angle based on the offset from the centre of the lane

        Args:
            image (List): 3D list representing an RGB image
            left_lane (AdvancedLine): Detected left-lane object (polynomial)
            right_lane (AdvancedLine): Detected right-lane object (polynomial)

        Returns:
            (Tuple): containing the image and the predicted steering angle
        """
        # calculate offset distance between centre of image and centre of lane
        offset_meters = self.compute_offset_from_center(left_lane, right_lane, image.shape[0])

        # update PID controller
        self.update_error(offset_meters)

        # predict steering
        steering_angle = self.predict_steering()

        return image, steering_angle

    def compute_offset_from_center(self, left_lane, right_lane, frame_width):
        """Computes an error offset from the centre of the lane in metres

        Args:
            left_lane (AdvancedLine): Polynomial line representing the left lane object
            right_lane (AdvancedLine): Polynomial line representing the right lane object
            frame_width (Integer): width of the frame

        Returns:

        """
        # only calculate the offset if both lanes have been detected
        if left_lane.detected and right_lane.detected:

            # get the mean of the x positions that have corresponding y positions that are concentrated white pixels in
            # both the left and right lanes
            left_lane_bottom = np.mean(left_lane.all_x[left_lane.all_y > 0.95 * left_lane.all_y.max()])
            right_lane_bottom = np.mean(right_lane.all_x[right_lane.all_y > 0.95 * right_lane.all_y.max()])

            # lane width is the difference between the mean x positions
            lane_width = right_lane_bottom - left_lane_bottom

            # calculate midpoint of the frame
            midpoint = frame_width / 2

            # the offset in pixels between the midpoint of the lane and the midpoint of the frame
            offset_pixels = (left_lane_bottom + (lane_width / 2) - midpoint)

            # get the offset in metres
            offset_meter = XM_PER_PIX * offset_pixels
        else:
            # no lanes detected
            offset_meter = -1

        return offset_meter
