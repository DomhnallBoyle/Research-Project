"""
    Filename: line/base_line.py
    Description: Contains functionality for an abstract base line that requires overriding
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
from abc import ABC, abstractmethod


class BaseLine(ABC):
    """Abstract base class for extension of simple and advanced line objects

    Contains generic abstract methods to be implemented by each line object

    Attributes:
        None
    """

    def __init__(self):
        """Abstract base class - cannot instantiate this class
        """
        pass

    @abstractmethod
    def draw(self, *args):
        """Abstract method to draw a line onto an image

        Args:
            *args (Tuple):

        Raises:
            NotImplementedError: if the method if not overridden in the derived class
        """
        raise NotImplementedError

    @abstractmethod
    def midpoint(self):
        """Abstract method to find the midpoint of a line

        Raises:
            NotImplementedError; if the method is not overridden in the derived class
        """
        raise NotImplementedError
