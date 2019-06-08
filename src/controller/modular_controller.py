"""
    Filename: controller/modular_controller.py
    Description: Contains functionality for the base controller base class
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

import argparse
import cv2
import math
import os
import sys

# for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from base_controller import BaseController
from lane_detection import *
from steer_controller import *


class ModularController(BaseController):
    """Class containing functionality for the modular approach to output steering angles given an image as input

    Attributes:
        lane_detection (BaseLineDetection): Module used to detect the lanes given an image
        steering_controller (SteeringController): Module to output a steering angle given 2 detected lanes
    """

    def __init__(self, lane_detection, steering_controller, debug=False):
        """Instantiate a instance of ModularController

        Calls __init__ of BaseController

        Args:
            lane_detection (BaseLineDetection): Module used to detect the lanes given an image
            steering_controller (SteeringController): Module to output a steering angle given 2 detected lanes
            debug (Boolean): for debugging purposes of the controller
        """
        super().__init__(debug)
        self.lane_detection = lane_detection
        self.steering_controller = steering_controller

    def get_steering_angle(self, image, roi_coords):
        """Implemented abstract method to get the steering angle given an image

        Run lane detection first and then get the steering angle from the detected lanes

        Args:
            image (List): 3D list representing an RGB image
            roi_coords (Tuple):

        Returns:
            (Float): steering angle in degrees
        """
        # run the lane detection using the image and region-of-interest coordinates
        left_lane, right_lane = self.lane_detection.detect_lanes(image, roi_coords)

        # if the left and right lanes are detected
        if left_lane and right_lane:
            # draw them onto the image
            left_lane.draw(image)
            right_lane.draw(image)

            # use the detected lanes to get the steering angle from the steering controller
            image, steering_angle = self.steering_controller.get_steering_angle(image, left_lane, right_lane)

            # reverse the angle. Left = positive, right = negative
            steering_angle = -steering_angle
        else:
            # lanes not detected, write this on this image
            cv2.putText(image, 'Lanes not detected!', (int(image.shape[1] / 2) - 300,
                                                       int(image.shape[0] / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # just go straight
            steering_angle = 0.0

        # show the image
        cv2.imshow('Modular Controller', image)

        # press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            exit(1)

        # return the steering angle in degrees
        return math.degrees(steering_angle)


def main(args):
    """Main method for creating the modular controller with modules depending on the arguments given through the CLI

    Outputs a steering angle using the lane-detection and steering controllers

    Args:
        args (Object): command line arguments

    Returns:
        None
    """
    # create the lane detection type using arguments from the CLI
    if args.lane_detection == 'simple':
        lane_detection = SimpleLaneDetection(debug=args.debug)
    else:
        lane_detection = AdvancedLaneDetection(debug=args.debug)

    # create the steering controller type usign arguments from the CLI
    if args.steer_controller == 'trivial':
        steering_controller = TrivialController()
    else:
        steering_controller = PIDController()

    # read in the image from the absolute image path
    image = cv2.imread(args.image_path, 1)

    # create the controller object using the lane detection and steering controller modules
    controller = ModularController(lane_detection, steering_controller, debug=args.debug)

    # get the steering angle from the image and ROI coords
    steering_angle = controller.get_steering_angle(image, args.horizon)

    print('Angle: {}'.format(steering_angle))


if __name__ == '__main__':
    """Entry point for the Python script

    Calls the main method with the command line arguments
    
    Example Usage: 
    python modular_controller.py <absolute_image_path> --lane_detection=simple --steer_controller=trivial --debug=True
    """
    arg_parser = argparse.ArgumentParser(description='Run a modular controller with an image')
    arg_parser.add_argument('image_path', type=str)
    arg_parser.add_argument('--lane_detection', default='simple', type=str)
    arg_parser.add_argument('--steer_controller', default='trivial', type=str)
    arg_parser.add_argument('--horizon', default='[[0, 600], [350, 400], [930, 400], [1280, 600]]', type=coords)
    arg_parser.add_argument('--debug', type=bool, default=False)

    main(arg_parser.parse_args())
