"""
    Filename: line/simple_line.py
    Description: Contains functionality for creating an instance of a simple straight line (y = mx + c)
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import cv2

# local source imports
from line.base_line import BaseLine


class SimpleLine(BaseLine):
    """Class that extends BaseLine that represents a simple straight line

    Attributes:
        slope (Float): gradient of the line (co-efficient)
        intercept (Float): intercept of the line on the y-axis (co-efficient)
        y1 (Integer): y point for the first coordinate
        y2 (Integer): y point for the second coordinate
        line_points (Tuple): contains a pair of (x, y) coordinates
    """

    def __init__(self, slope, intercept, y1, y2):
        """Instantiating an instance of SimpleLine

        Calls the __init__ of the base class BaseLine

        Args:
            slope (Float): gradient of the line (co-efficient)
            intercept (Float): intercept of the line on the y-axis (co-efficient)
            y1 (Integer): y point for the first coordinate
            y2 (Integer): y point for the second coordinate
        """
        super().__init__()
        self.slope = slope
        self.intercept = intercept
        self.y1, self.y2 = y1, y2
        self.line_points = self.make_line_points()

    def draw(self, image):
        """Implemented abstract method from the base class

        Draw the detected line on the image

        Args:
            image (List): 3D list representing an RGB image

        Returns:
            None
        """
        cv2.line(image, self.line_points[0], self.line_points[1], (0, 0, 255), 5)
        cv2.line(image, self.line_points[0], self.line_points[1], (0, 0, 255), 5)

    def make_line_points(self):
        """Calculate the 2 points that make up the ends of the line

        Use the simple line equation y = mx + c to calculate the x points

        Raises:
            OverflowError: if the arithmetic operation has exceeded the limits of the current Python runtime

        Returns:
            (Tuple): Pair of (x, y) coordinates representing the points of a line (2 ends)
        """
        try:
            # x = (y - c) / m
            x1 = int((self.y1 - self.intercept) / self.slope)
            x2 = int((self.y2 - self.intercept) / self.slope)
            y1 = int(self.y1)
            y2 = int(self.y2)
        except OverflowError:
            return None

        return (x1, y1), (x2, y2)

    def midpoint(self):
        """Implemented abstract method from the base class

        Calculate the midpoint of a line given the line points
        The midpoint is the average of the xs and ys between a pair of coordinates

        Returns:
            (Tuple): (x, y) midpoint coordinate
        """
        # get the 2 (x, y) line points
        p1 = self.line_points[0]
        p2 = self.line_points[1]

        # calculate average between 2 points
        return int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
