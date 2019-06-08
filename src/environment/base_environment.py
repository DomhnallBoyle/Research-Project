"""
    Filename: environment/base_environment.py
    Description: Contains functionality for the base environment that contains a specific controller
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
from abc import ABC
import sys
import os

# for local source imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# local source imports
from controller import *
from lane_detection import *
from steer_controller import *


class BaseEnvironment(ABC):
    """Abstract class representing a base environment for the wheelchair and CARLA environments

    Generic functions and attributes are included in this class

    Attributes:
        controller (BaseController): type of controller the environment used (modular/e2e)
    """

    def __init__(self):
        """Abstract base class - cannot instantiate instance of abstract class
        """
        self.controller = None

    def get_controller(self, args):
        """Generic function to obtain a controller based on the arguments given from the CLI

        Args:
            args (Object): command line arguments

        Returns:
            None
        """
        # depending on type of controller i.e. modular/e2e
        if args.controller == 'modular':

            # lane detection type
            if args.lane_detection == 'simple':
                lane_detection = SimpleLaneDetection(debug=args.debug)
            else:
                lane_detection = AdvancedLaneDetection(debug=args.debug)

            # steering controller type
            if args.steer_controller == 'trivial':
                steering_controller = TrivialController()
            elif args.steer_controller == 'edt':
                steering_controller = EDTController()
            elif args.steer_controller == 'pid':
                steering_controller = PIDController()

            # controller object
            self.controller = ModularController(lane_detection, steering_controller, debug=args.debug)

        else:
            # NOTE: Won't be able to run CARLA with any trained models because of amount of VRAM needed to run CARLA and
            # model simultaneously
            # picking the appropriate model using the CLU arguments
            if args.model_type == 'nvidia':
                model = NvidiaModel(args.model_path)
            else:
                model = None

            self.controller = E2EController(model)
