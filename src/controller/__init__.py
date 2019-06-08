"""
    Filename: controller/__init__.py
    Description: Contains functionality for importing all of the classes within this directory for use by other modules
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import sys
import os

# for local source imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# local source imports
from e2e_controller import *
from modular_controller import *
