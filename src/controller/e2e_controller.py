"""
    Filename: controller/e2e_controller.py
    Description: Contains functionality for the base controller base class
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# local source imports
from base_controller import BaseController


class E2EController(BaseController):
    """Class that represents an End-to-end controller

    Contains functionality for sending controls using end-to-end deep learning models

    Attributes:
        model (BaseModel): Reference to the end-to-end model that the controller uses
    """

    def __init__(self, model):
        """Instantiate an instance of the E2EController class

        Calls the __init__ of the BaseController class

        Args:
            model (BaseModel): End-to-end model that the controller uses
        """
        super().__init__()
        self.model = model

    def get_steering_angle(self, image):
        """Implemented abstract method that retrieves a steering angle from a trained CNN

        Args:
            image (List): 3D list representing an RGB image

        Returns:
            (Float): predicted steering angle from the trained e2e model
        """
        return self.model.test_model(image)
