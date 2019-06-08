"""
    Filename: utils/wheelchair_controller.py
    Description: Contains functionality for manually and automatically controlling the wheelchair
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import argparse
import cv2
import os
import pickle
import pygame
import requests
import sys
import time
from struct import pack
from threading import Thread

# for local source imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# local source imports
from environment import *
from joystick import WheelchairJoystick
from lane_detection import *
from sockets import *
from utils import *

# global constants
WHEELCHAIR_URL = 'http://xavier.local/scripts/serialSend.php?serialData={},{},{}'
MAX_SPEED = 30

# global variables
global_image = None
global_steer = 0.0


class WheelchairControllerClientProtocol(ClientProtocol):
    """Class that runs in a thread collecting the steering angles obtained from the processing node

    Attributes:
        None
    """

    def __init__(self):
        """Instantiating an instance of WheelchairControllerClientProtocol

        Calls the __init__ of the superclass ClientProtocol
        """
        super().__init__()

    def handle_function(self, data):
        """Overridden method that acts on the data received from the processing node

        Args:
            data (Dict): containing the steering angle

        Returns:
            None
        """
        steering_angle = data['steer']

        # sets the global steer variable with the steering angle received
        global global_steer
        global_steer = steering_angle

        print(global_steer)


class WheelchairControllerServerProtocol(ServerProtocol):
    """Class that sends the collected image from the centre camera to the node for processing

    Attributes:
        None
    """

    def __init__(self):
        """Instantiating an instance of WheelchairControllerServerProtocol

        Calls the __init__ of the superclass ServerProtocol
        """
        super().__init__()

    def handle_function(self):
        """Overridden method that continuously loops, sending data to the processing node

        Args:
            None

        Returns:
            None
        """
        while True:
            # package the data to send in a dictionary
            data = {
                'image': global_image
            }

            # use struct to make sure we have a consistent endianness on the length
            length = pack('>Q', len(pickle.dumps(data)))

            # sendall to make sure it blocks if there's back-pressure on the socket
            # send the data length first to ensure the receiving end knows the amount being sent
            self.socket.sendall(length)

            # send the pickled data
            self.socket.sendall(pickle.dumps(data))

            # to handle response from sent data
            ack = self.socket.recv(1)


class WheelchairController(BaseEnvironment):
    """Class that is responsible for manually and automatically controlling the wheelchair

    Attributes:
        args (Object): command line arguments
        control_type (String): determines whether to use manual or automatic controlling
        debug (Boolean): for debugging purposes
    """

    def __init__(self, args):
        """Instantiating an instance of WheelchairController

        Calls the appropriate running type based on the command line arguments

        Args:
            args: (Object) command line arguments
        """
        self.args = args
        self.control_type = args.control_type
        self.debug = args.debug

        # calls the appropriate class method based on the control type
        getattr(self, self.control_type)()

    def control_wheelchair(self, throttle):
        """Method for constantly controlling the wheelchair.

        This method runs in a thread calling the wheelchair API.

        Args:
            throttle (Integer): speed to use for the wheelchair

        Returns:
            None
        """
        while True:
            self.call_api(throttle, global_steer, 'RUN')

    def manual(self):
        """Method for manually controlling the wheelchair

        Returns:
            None
        """
        # create an instance of the wheelchair joystick
        joystick = WheelchairJoystick()

        while True:
            # continuously loop until done

            for event in pygame.event.get():
                # loop through events, if window shut down, quit program
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # get the angle and speed from the controller
            angle, speed = joystick.get_controls(max_speed=args.max_speed)

            # call the wheelchair API with these controls
            self.call_api(speed, angle, 'RUN')

            # if stop pressed, perform an emergency stop
            if joystick.stop_pressed():
                self.emergency_stop()

    def automatic(self):
        """Method for automatically controlling the wheelchair

        Returns:
            None
        """
        # USAGE (center camera): nc -l -p 5000 | python wheelchair.py automatic

        # uncomment when ready
        # client = WheelchairControllerClientProtocol(args)
        # client.listen('0.0.0.0', 9999)

        # server = WheelchairControllerServerProtocol(args)
        # server.connect('192.168.1.103', 9999)

        # create an instance of the video stream and start the thread
        fvs = FileVideoStream('/dev/stdin').start()

        # get the correct method of controlling the wheelchair i.e. modular or end-to-end
        self.get_controller(self.args)

        # countdown from 10 before starting for setting up reasons
        count = 0
        while True:
            print('Starting in {} seconds'.format(10-count))
            fvs.clear()
            time.sleep(1)
            count += 1

            if count == 10:
                break

        # start the thread to constantly call the wheelchair API with a throttle speed
        controller_thread = Thread(target=self.control_wheelchair, args=(self.args.throttle, ))
        controller_thread.start()

        while True:
            # continously read frames from the centre camera video stream
            frame = fvs.read()

            # flip the images because they're upside down
            frame = cv2.flip(frame, -1)

            # set the global image variable allowing the server socket to send the image data
            global global_image
            global_image = frame

            # clear the video stream for the next camera image
            # this removes any outdated images - gets the most current image
            fvs.clear()

    def call_api(self, speed, angle, action):
        """Function for calling the wheelchair API with specific controls

        Args:
            speed (Integer): speed to control the wheelchair with
            angle (Integer): angle of direction to steer
            action (String): action to perform, 'RUN' or 'STOP'

        Returns:
            None
        """
        print('Steer: {}, Throttle: {}, Action: {}'.format(angle, speed, action))

        # make a POST request if not in debug mode
        if not self.debug:
            requests.post(WHEELCHAIR_URL.format(speed, angle, action))

    def emergency_stop(self):
        """Perform an emergency stop function. Calls the API with the 'STOP' action

        Exits the Python script after completion

        Returns:
            None
        """
        print('Emergency stop sent, exiting...')
        self.call_api(0, 0, 'STOP')

        exit(0)


def main(args):
    """Main function for creating an instance of the WheelchairController

    Args:
        args (Object): command line arguments

    Returns:
        None
    """
    # exit the script if an incorrect command is used
    if args.control_type:
        WheelchairController(args)
    else:
        print('Please enter a control type: \'manual\' or \'automatic\'')


if __name__ == '__main__':
    """Main starting point for the Python script, parses the command line arguments
    
    Calls the main method with the command line arguments
    
    Example Usage: 
    python wheelchair_controller.py manual --max_speed=20 --debug=True
    
    python wheelchair_controller.py automatic --debug=True
    """
    parser = argparse.ArgumentParser('Wheelchair Control')
    subparsers = parser.add_subparsers(dest='control_type', help='sub-command help')

    parser_training = subparsers.add_parser('manual', help='Manual control')
    parser_training.add_argument('--max_speed', default=MAX_SPEED, type=int)
    parser_training.add_argument('--debug', default=False)

    parser_testing = subparsers.add_parser('automatic', help='Automatic control')
    parser_testing.add_argument('--debug', default=False)

    args = parser.parse_args()

    main(args)
