"""
    Filename: utils/drawing.py
    Description: Contains functionality for drawing line points onto an image frame
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import cv2
import math

# global constants
RADIANS_90 = math.radians(90)


def get_line_points(image_width, image_height, angle):
    """Creates 2 (x, y) line points

    1 is located at the bottom of the image and the other connecting line point is located halfway up the image
    depending on the line angle

    x2 = x1 + (d * cos(angle))
    y2 = y1 + (d * sin(angle))
    where d = line distance

    Args:
        image_width (Integer): width of the image
        image_height (Integer): height of the image
        angle (Float): angle between the line and x-axis

    Returns:
        (Tuple): 2 (x, y) points on the image
    """
    # this point is located at the bottom the the image, halfway
    point_1 = (int(image_width / 2), int(image_height))

    # this point is located halfway up the image
    point_2 = (
        int(point_1[0] + int(image_height / 2) * math.cos(angle)),
        int(point_1[1] + int(image_height / 2) * math.sin(angle))
    )

    return point_1, point_2


def draw_on_frame(frame, prediction, groundtruth, put_text=True):
    """Draw the predicted and groundtruth lines on the image frame

    Args:
        frame (List): 3D list representing an RGB image
        prediction (Float): predicted angle from the controller
        groundtruth (Float): groundtruth angle
        put_text (Boolean): whether to add text to the image or not

    Returns:
        (List): 3D image with the lines drawn on the frame
    """
    # get the angles between the line and the x-axis of the image
    line_angle_prediction = -RADIANS_90 - prediction
    line_angle_groundtruth = -RADIANS_90 - groundtruth

    # get the width and height of the frame
    h, w = frame.shape[:2]

    # get the (x, y) coordinate points for the lines
    prediction_point_1, prediction_point_2 = get_line_points(w, h, line_angle_prediction)
    groundtruth_point_1, groundtruth_point_2 = get_line_points(w, h, line_angle_groundtruth)

    # prediction is red, groundtruth is green
    cv2.line(frame, prediction_point_1, prediction_point_2, (0, 0, 255), 2)
    cv2.line(frame, groundtruth_point_1, groundtruth_point_2, (0, 255, 0), 2)

    # add the predicted and groundtruth labels to the image if True
    if put_text:
        cv2.putText(frame,
                    'Predicted: {}'.format(round(math.degrees(prediction), 3)),
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 0))

        cv2.putText(frame,
                    'Groundtruth: {}'.format(round(math.degrees(groundtruth), 3)),
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2, (255, 0, 0))

    return frame
