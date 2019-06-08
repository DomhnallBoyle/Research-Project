"""
    Filename: lane_detection/advanced_lane_detection.py
    Description: Contains functionality for detecting advanced polynomial lanes from an image
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
from line.advanced_line import AdvancedLine
from lane_detection.utils.binarisation import *
from lane_detection.utils.general import *
from lane_detection.utils.perspective import *

# global constants
MIN_YELLOW_THRESHOLD = np.array([0, 70, 70])
MAX_YELLOW_THRESHOLD = np.array([50, 255, 255])
ym_per_pix = 30 / 720   # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension


class AdvancedLaneDetection(BaseLaneDetection):
    """Class for running the advanced lane detection algorithm for detecting curved lines on an image

    Extends the abstract lane detection class

    Attributes:
        None
    """

    def __init__(self, debug=False):
        """Initialise an instance of the AdvancedLineDetection

        Calls __init__ of the base class BaseLaneDetection

        Based off the work from the udacity self-driving program:
        https://github.com/ndrplz/self-driving-car/tree/master/project_4_advanced_lane_finding

        Args:
            debug (Boolean): for debugging the lane detection algorithm
        """
        super().__init__(debug)

    def detect_lanes(self, image, roi_coords):
        """Implemented abstract method to return the detected lane objects from the image

        Acts as a wrapper for the process of lane detection; involves pre-processing

        Args:
            image (List): 3D list representing an RGB image
            roi_coords (List): contains tuples of coordinates used to extract the region-of-interest from the image

        Returns:
            (Tuple): contins the AdvancedLine objects representing the detected lines
        """
        # apply pre-processing techniques
        preprocessed_image, transform_inversed = self.preprocess(image, roi_coords)

        # get the lane objects
        left_lane, right_lane = self.get_lanes(preprocessed_image)

        return left_lane, right_lane

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

        # find yellow lines
        yellow_threshold_image = threshold_HSV_image(image, MIN_YELLOW_THRESHOLD, MAX_YELLOW_THRESHOLD)

        # find white lines >= 250
        white_threshold_image = threshold_equalised_image(image, threshold=250)

        # combine yellow and white lines
        combined_lines = np.logical_or(yellow_threshold_image, white_threshold_image)

        # apply sobel edge detection
        sobel_edges = threshold_sobel_image(image, threshold=50, kernel_size=9)

        # combine lines with sobel edges
        combined_lines_and_edges = np.logical_or(combined_lines, sobel_edges)

        # apply CLOSED morphology - Dilation and then Erosion
        binary = apply_morphology(combined_lines_and_edges, kernel_size=5, morphology_type=cv2.MORPH_CLOSE)

        # get a region of interest
        roi = get_roi(binary, roi_coords)

        # get the birds eye view from roi
        birds_eye, transform_inversed = get_birds_eye(roi, roi_coords)

        # show the images if in debug mode
        if self.debug:
            images = [binary, roi, birds_eye]
            titles = ['Binary', 'ROI', 'Birds Eye']
            cv2_show_figured_images(images=images, titles=titles)

        return birds_eye, transform_inversed

    def get_lanes(self, image):
        """Apply the sliding window technique to get the lane lines from the pre-processed image

        Args:
            image (List): 2D list representing a pre-processed grayscale image

        Returns:
            (Tuple):
        """
        # create the advanced lane objects
        left_lane = AdvancedLine()
        right_lane = AdvancedLine()

        # find lane line pixels and fit line positions with polynomials
        left_lane, right_lane, img_fit, found = self.get_fits_by_sliding_windows(image, left_lane, right_lane)

        return left_lane, right_lane

    def get_fits_by_sliding_windows(self, birdeye_binary, line_lt, line_rt, n_windows=9, verbose=False):
        """Get polynomial coefficients for lane-lines detected in an binary image.

        Uses a vertical sliding window technique from the base of the image

        Args:
            birdeye_binary (List): 2D input bird's eye view binary image
            line_lt (AdvancedLine): left lane-line previously detected
            line_rt (AdvancedLine): right lane-line previously detected
            n_windows (Integer): number of sliding windows used to search for the lines
            verbose (Boolean): if True, display intermediate output

        Returns:
            (Tuple): containing the updates lane lines and ouput image
        """
        # get the width and height of the image
        height, width = birdeye_binary.shape

        # assuming you have created a warped binary image called "binary_warped"
        # take a histogram of the bottom half of the image
        histogram = np.sum(birdeye_binary[height // 2:-30, :], axis=0)

        # create an output image to draw on and visualize the result
        out_img = np.dstack((birdeye_binary, birdeye_binary, birdeye_binary)) * 255

        # find the peak of the left and right halves of the histogram
        # these will be the starting point for the left and right lines
        midpoint = len(histogram) // 2
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # set height of windows
        window_height = np.int(height / n_windows)

        # identify the x and y positions of all nonzero pixels in the image
        nonzero = birdeye_binary.nonzero()
        nonzero_y = np.array(nonzero[0])
        nonzero_x = np.array(nonzero[1])

        # current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        margin = 100  # width of the windows +/- margin
        minpix = 50  # minimum number of pixels found to recenter window

        # create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # step through the windows one by one
        for window in range(n_windows):
            # identify window boundaries in x and y (and right and left)
            win_y_low = height - (window + 1) * window_height
            win_y_high = height - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            # draw the windows on the visualization image
            cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)

            # identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xleft_low)
                              & (nonzero_x < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzero_y >= win_y_low) & (nonzero_y < win_y_high) & (nonzero_x >= win_xright_low)
                               & (nonzero_x < win_xright_high)).nonzero()[0]

            # append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # if you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzero_x[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzero_x[good_right_inds]))

        # concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        # extract left and right line pixel positions
        line_lt.all_x, line_lt.all_y = nonzero_x[left_lane_inds], nonzero_y[left_lane_inds]
        line_rt.all_x, line_rt.all_y = nonzero_x[right_lane_inds], nonzero_y[right_lane_inds]

        # assume detected until aren't
        detected = True
        if not list(line_lt.all_x) or not list(line_lt.all_y):
            # if no detected right pixels coords, use previous co-efficients
            left_fit_pixel = line_lt.last_fit_pixel
            left_fit_meter = line_lt.last_fit_meter
            detected = False
        else:
            # create new co-efficients if both (x, y) pixel coords
            left_fit_pixel = np.polyfit(line_lt.all_y, line_lt.all_x, 2)
            left_fit_meter = np.polyfit(line_lt.all_y * ym_per_pix, line_lt.all_x * xm_per_pix, 2)

        if not list(line_rt.all_x) or not list(line_rt.all_y):
            # if no detected left pixels coords, use previous co-efficients
            right_fit_pixel = line_rt.last_fit_pixel
            right_fit_meter = line_rt.last_fit_meter
            detected = False
        else:
            # create new co-efficients if both (x, y) pixel coords
            right_fit_pixel = np.polyfit(line_rt.all_y, line_rt.all_x, 2)
            right_fit_meter = np.polyfit(line_rt.all_y * ym_per_pix, line_rt.all_x * xm_per_pix, 2)

        # return if
        if left_fit_pixel is None or right_fit_pixel is None:
            return line_lt, line_rt, out_img, False

        # update the co-efficients and detected flag
        line_lt.update_line(left_fit_pixel, left_fit_meter, detected=detected)
        line_rt.update_line(right_fit_pixel, right_fit_meter, detected=detected)

        # generate x and y values for plotting
        ploty = np.linspace(0, height - 1, height)
        left_fitx = left_fit_pixel[0] * ploty ** 2 + left_fit_pixel[1] * ploty + left_fit_pixel[2]
        right_fitx = right_fit_pixel[0] * ploty ** 2 + right_fit_pixel[1] * ploty + right_fit_pixel[2]

        # blue for left lane, red for right lane (BGR)
        # apply these colours to the indexes of the detected non-zero x and y pixels
        out_img[nonzero_y[left_lane_inds], nonzero_x[left_lane_inds]] = [255, 0, 0]
        out_img[nonzero_y[right_lane_inds], nonzero_x[right_lane_inds]] = [0, 0, 255]

        # display the results if in debug mode
        if verbose:
            # display results
            images = [birdeye_binary, out_img]
            titles = ['Before', 'After']
            cv2_show_figured_images(images=images, titles=titles)

        return line_lt, line_rt, out_img, True


def main(args):
    """Main function for creating an instance of the AdvancedLaneDetection algorithm and running it on a video

     Args:
         args (Object): contains the command line arguments

     Returns:
         None
     """
    # construct the video object using the absolute file path
    video = cv2.VideoCapture(args.video_file)

    # advanced lane detection algorithm for the video
    advanced_lane_detection = AdvancedLaneDetection(debug=args.debug)

    while video.isOpened():
        # continue reading frames from the video
        success, frame = video.read()
        if success:

            # detect the left and right lanes from the frame
            left_lane, right_lane = advanced_lane_detection.detect_lanes(frame, args.coords)

            # if both lanes detected
            if left_lane and right_lane:
                # draw them onto the frame
                left_lane.draw(frame)
                right_lane.draw(frame)
            else:
                # lanes not detected
                cv2.putText(frame, 'Lanes not detected!', (int(WIDTH / 2) - 300,
                                                           int(HEIGHT / 2)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # show the image
            cv2.imshow('Advanced Lane Detection', frame)

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
    python advanced_lane_detection.py <video_file_path> '[[0, 720], [0, 0], [1280, 0], [1280, 720]]' --debug=True
    """
    parser = argparse.ArgumentParser(description='Advanced lane detection.')
    parser.add_argument('video_file', help='Absolute path to video.', type=str)
    parser.add_argument('coords', help='ROI horizon coords', type=coords)
    parser.add_argument('--debug', help='Debug mode.', type=bool, default=False)
    args = parser.parse_args()

    main(args)
