"""
    Filename: environment/wheelchair.py
    Description: Wheelchair environment for processing the images using the controller and returning steering angles
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
import pickle
from struct import pack
import sys

# for local source imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# local source imports
from environment import *
from sockets import *

# global variables
global_steer = 0.0


class WheelchairClientProtocol(ClientProtocol, BaseEnvironment):
    """Class that extends the client socket and BaseEnvironment

    Attributes:
        args (Object): command line arguments
    """

    def __init__(self, args):
        """Instantiate an instance of the WheelchairClientProtocol

        Calls the __init__ of the extended classes
        Get the controller based on the command line arguments
        """
        super().__init__()
        self.args = args
        # get the controller using the command line arguments
        self.get_controller(args)

    def handle_function(self, data):
        """Function that receives the centre image from the stream and uses the controller to output a steering angle

        Args:
            data (Dictionary): dictionary of data from the server

        Returns:
            None
        """
        # get the image data from the dictionary
        image = data['image']

        # decode the image back to its original form from a byte string
        image = cv2.imdecode(np.asarray(bytearray(image), dtype=np.uint8), 1)

        # use the global variable to get the steering angle using the controller
        global global_steer
        global_steer = self.controller.get_steering_angle(image, args.horizon)


class WheelchairServerProtocol(ServerProtocol):
    """Class that extends the server socket and provides functionality for sending the processed steering angle back
    to the machine that sent the image
    """

    def __init__(self, args):
        """Instantiate an instance of the WheelchairServerProtocol

        Calls the __init__ of the extended classes

        """
        super().__init__()
        self.args = args

    def handle_function(self):
        """Function to pack the steering angle from the controller and send it back to the machine that sent the
        image to be processed

        Continuously loops sending back steering angles from the global variable

        Returns:
            None
        """
        while True:
            # pack the data into a dictionary
            data = {
                'steer': global_steer
            }

            # use struct to make sure we have a consistent endianness on the length
            length = pack('>Q', len(pickle.dumps(data)))

            # sendall to make sure it blocks if there's back-pressure on the socket
            self.socket.sendall(length)
            self.socket.sendall(pickle.dumps(data))

            # receive the success token
            ack = self.socket.recv(1)


def main(args):
    """Main method to setup the client-server style architecture

    Client receives the images from the centre camera
    Server sends back the steering angle from the controller

    Args:
        args (Object): command line arguments

    Returns:
        None
    """
    # server on other machine needs to connect to this machines IP
    client = WheelchairClientProtocol(args)
    client.listen('0.0.0.0', 9999)

    # server on this machines needs to connect to other machines IP
    server = WheelchairServerProtocol(args)
    server.connect('192.168.1.106', 9999)


if __name__ == '__main__':
    """Entry point for the Python script
    
    Calls the main method with the command line arguments
    
    Example Usage: 
    python wheelchair.py
    """
    argparser = argparse.ArgumentParser(description='Wheelchair processing Client-Server')

    args = argparser.parse_args()

    main(args)
