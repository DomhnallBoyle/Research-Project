"""
    Filename: lane_detection/utils/calibration.py
    Description: Contains functionality for finding the calibration of chessboard images
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import cv2
import glob
import numpy as np
import os
import pickle


def calibrate_camera(verbose=False):
    """Function for finding the calibration co-efficients from chessboard images

    Required to calibrate the images from the camera images

    Args:
        verbose (Boolean): for debugging purposes

    Returns:
        (Dictionary): containing the calibration co-efficients
    """
    current_directory = os.path.abspath(os.path.dirname(__file__))
    calibration_cache = os.path.join(current_directory, 'calibration_images/calibration_data.pickle')

    # if the calibration co-efficients already exist, load them and return them
    if os.path.exists(calibration_cache):
        with open(calibration_cache, 'rb') as dump_file:
            calibration = pickle.load(dump_file)

            return calibration
    else:
        # they don't exist, calculate them

        # 3d points in real world space
        object_points = []

        # 2d points in image plane.
        image_points = []

        # number of inside corners of the chessboard images in the x and y axis
        num_x_inside_corners = 9
        num_y_inside_corners = 6

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)
        object_point = np.zeros((num_x_inside_corners * num_y_inside_corners, 3), np.float32)
        object_point[:, :2] = np.mgrid[0:num_x_inside_corners, 0:num_y_inside_corners].T.reshape(-1, 2) # x, y coordinate

        # for all the chessboard images in the directory
        for filename in glob.glob('calibration_images/calibration*.jpg'):
            # read the image and convert to grayscale
            image = cv2.imread(filename)
            grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # find the corners from the image
            success, corners = cv2.findChessboardCorners(grayscale_image, (num_x_inside_corners, num_y_inside_corners),
                                                         None)

            if success:
                # append the real and image coords to the lists
                image_points.append(corners)
                object_points.append(object_point)

                # display the image if in debugging mode
                if verbose:
                    cv2.drawChessboardCorners(image, (num_x_inside_corners, num_y_inside_corners), corners, success)
                    cv2.imshow('Image', image)
                    cv2.waitKey(500)

        # destroy all windows if in debug mode
        if verbose:
            cv2.destroyAllWindows()

        # get the image dimensions
        image_size = (image.shape[1], image.shape[0])

        # get the calibration co-efficients using the real-world coords and coords from the image of the corners
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

        # 3x3 floating-point camera matrix
        # vector of distortion coefficients
        calibration = {'mtx': mtx, 'dist': dist}

        # pickle and save the co-efficients to file
        with open(calibration_cache, 'wb') as dump_file:
            pickle.dump(calibration, dump_file)

        return calibration
