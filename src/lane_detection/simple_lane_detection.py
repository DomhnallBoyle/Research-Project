"""
    Filename: lane_detection/simple_lane_detection.py
    Description: Contains functionality for detecting simple lanes from an image
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import argparse
import cv2
import numpy as np
import os
import sys

# for local source imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# local source imports
from lane_detection.base_lane_detection import BaseLaneDetection
from line.simple_line import SimpleLine
from lane_detection.utils.draw import *
from lane_detection.utils.general import *
from lane_detection.utils.perspective import get_roi


class SimpleLaneDetection(BaseLaneDetection):
    """Class for running the simple lane detection algorithm for detecting straight lane line in an image

    Extends the abstract lane detection class

    Attributes:
        None
    """

    def __init__(self, debug=False):
        """Initialise an instance of the SimpleLaneDetection

        Calls __init__ of the the base class BaseLaneDetection

        Based off the work from:
        https://github.com/ndrplz/self-driving-car/tree/master/project_1_lane_finding_basic
        https://github.com/naokishibuya/car-finding-lane-lines
        https://github.com/Sentdex/pygta5/blob/master/Tutorial%20Codes/Part%201-7/part-7-self-driving-example.py

        Args:
            debug (Boolean): for debugging the lane detection algorithm
        """
        super().__init__(debug)

    def detect_lanes(self, image, roi_coords):
        """Implemented abstract method to return the detected lane objects from the image

        Acts as a wrapper for the process of lane detection; involves pre-processing;

        Args:
            image (List): 3D list representing an RGB image
            roi_coords (List): contains tuples of coordinates used to extract the region-of-interest from the image

        Returns:
            (Tuple): contains the SimpleLine objects representing the detected lanes
        """
        # apply pre-processing techniques
        preprocessed_image = self.preprocess(image, roi_coords)

        # get the lane objects
        lanes = self.get_lanes(preprocessed_image)

        return lanes

    def preprocess(self, image, roi_coords):
        """Implemented abstract method for applying specific pre-processing techniques

        Args:
            image (List): 3D list representing an RGB image
            roi_coords (List): contains tuples of coordinates used to extract the region-of-interest from the image

        Returns:
            (List): 2D list representing a pre-processed grayscale image
        """
        # resize
        image = resize(image)

        # undistort image
        image = cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

        # convert to HLS
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

        # white color mask
        lower = np.uint8([0, 150, 0])
        upper = np.uint8([255, 255, 255])
        white_mask = cv2.inRange(hls, lower, upper)

        # yellow color mask
        lower = np.uint8([10, 0, 100])
        upper = np.uint8([40, 255, 255])
        yellow_mask = cv2.inRange(hls, lower, upper)

        # combine the mask
        mask = cv2.bitwise_or(white_mask, yellow_mask)
        image = cv2.bitwise_and(image, image, mask=mask)

        # convert to grayscale
        grayscale = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # apply blurring
        smoothing = cv2.GaussianBlur(grayscale, (15, 15), 0)

        # canny edge detection
        canny = cv2.Canny(smoothing, 50, 150)

        # get roi
        roi = get_roi(canny, roi_coords)

        # show the images if in debug mode
        if self.debug:
            images = [grayscale, smoothing, canny, roi]
            titles = ['Grayscale', 'Smoothing', 'Canny', 'ROI']
            cv2_show_figured_images(images=images, titles=titles)

        return roi

    def get_lanes(self, image, threshold=20, min_length=100, max_gap=300):
        """Apply Hough Transform to get the lane lines from the pre-processed image

        Args:
            image (List): 2D list presenting a pre-processed grayscale image
            threshold (Integer): number of votes needed to return lines
            min_length (Integer): minimum length of the line, rejected if smaller
            max_gap (Integer): maximum allowed gap between points on the same line

        Returns:
            (Tuple): containing the left and right SimpleLine objects
        """
        # apply the probabilistic hough transform algorithm using specific line settings
        lines = cv2.HoughLinesP(image, rho=1, theta=np.pi / 180, threshold=threshold, minLineLength=min_length,
                                maxLineGap=max_gap)

        # if lines detected from hough transform, construct the SimpleLine objects
        if lines is not None:
            return self.lane_lines(image, lines)

        return None, None

    def lane_lines(self, image, lines):
        """Constructs the SimpleLine objects from the calculated slope/intercept

        Executed on lines found from Hough Transform

        Args:
            image (List): 2D list representing a pre-processed grayscale image
            lines (List): list containing list of [x1, y1, x2, y2] coords

        Returns:
            (Tuple): containing constructed lane line objects (SimpleLine)
        """
        # get the average slope/intercept for left and right lane
        left_lane_si, right_lane_si = self.average_slope_intercept(lines)

        # y coords for how long the line should be across the image
        y1 = image.shape[0]
        y2 = y1 * 0.1

        # variables to hold line objects
        left_lane, right_lane = None, None

        # construct the SimpleLine objects from the slope/intercept and y points
        if left_lane_si is not None and right_lane_si is not None:
            left_lane = SimpleLine(left_lane_si[0], left_lane_si[1], y1, y2)
            right_lane = SimpleLine(right_lane_si[0], right_lane_si[1], y1, y2)

        return left_lane, right_lane

    def average_slope_intercept(self, lines):
        """Function to calculate the average slope and intercept from lines found from Hough Transform

        Args:
            lines (List): list of [x1, y1, x2, y2] coords

        Returns:
            (Tuple): (slope, intercept) for each lane
        """
        left_lines = []  # (slope, intercept)
        left_weights = []  # (length,)
        right_lines = []  # (slope, intercept)
        right_weights = []  # (length,)

        # for each line coords
        for line in lines:

            # extract the points from the line
            for x1, y1, x2, y2 in line:

                # ignore vertical lines
                if x2 == x1:
                    continue

                # calculate the slope and intercept
                slope = (y2 - y1) / (x2 - x1)
                intercept = y1 - slope * x1

                # length of line - euclidean distance
                length = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

                # y is reversed in image
                if slope < 0:
                    # negative gradient (\)
                    left_lines.append((slope, intercept))
                    left_weights.append((length))
                else:
                    # positive gradient (/)
                    right_lines.append((slope, intercept))
                    right_weights.append((length))

        # add more weight to longer lines
        # get the average slope/intercept for each lane
        left_lane = np.dot(left_weights, left_lines) / np.sum(left_weights) if len(left_weights) > 0 else None
        right_lane = np.dot(right_weights, right_lines) / np.sum(right_weights) if len(right_weights) > 0 else None

        # (slope, intercept), (slope, intercept)
        return left_lane, right_lane


def main(args):
    """Main function for creating an instance of the SimpleLaneDetection algorithm and running it on a video

    Args:
        args (Object): contains the command line arguments

    Returns:
        None
    """
    # construct the video object using the absolute file path
    video = cv2.VideoCapture(args.video_file)

    # simple lane detection algorithm for the video
    simple_lane_detection = SimpleLaneDetection(debug=args.debug)

    while video.isOpened():
        # continue reading frames from the video
        success, frame = video.read()
        if success:

            # detect the left and right lanes from the frame
            left_lane, right_lane = simple_lane_detection.detect_lanes(frame, args.coords)

            # if both lanes detected
            if left_lane and right_lane:
                # draw them onto the frame
                left_lane.draw(frame)
                right_lane.draw(frame)
            else:
                # lanes not detected
                cv2.putText(frame, 'Lanes not detected!', (int(WIDTH / 2) - 300, int(HEIGHT / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # show the image
            cv2.imshow('Simple Lane Detection', frame)

            # press Q on keyboard to exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break

    # release the video capture and destroy openCV windows
    video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    """Main starting point for the Python script, parses the command line arguments
    
    Calls the main method with the command line arguments
    
    Example Usage: 
    python simple_lane_detection.py <video_file_path> '[[0, 720], [0, 0], [1280, 0], [1280, 720]]' --debug=True
    """
    parser = argparse.ArgumentParser(description='Simple lane detection using Hough Transform.')
    parser.add_argument('video_file', help='Absolute path to video.', type=str)
    parser.add_argument('coords', help='ROI horizon coords', type=coords)
    parser.add_argument('--debug', help='Debug mode.', type=bool, default=False)
    args = parser.parse_args()

    main(args)
