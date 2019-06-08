"""
    Filename: utils/capture_data.py
    Description: Contains functionality for capturing the wheelchair images and PS3 controller steering angles
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import argparse
import cv2
import math
import os
import pygame
import shutil
import sys
import time

# for local source imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# local source imports
from joystick import WheelchairJoystick
from utils import FileVideoStream

# global constants
IMAGE_FILENAME = '{}-{}.jpg'


class CaptureData:
    """Class for capturing the images from the camera streams and angles from the PS3 controller for saving to a
    driving log

    Attributes:
        args (Object): command lines arguments
        run_type (String): Whether to run the capturing in "debug" or "capture" mode
        camera_position (String): Either "Centre", "Left" or "Right" - for image file names
        joystick (WheelchairJoystick): Joystick to control the wheelchair - groundtruth angles extracted from this
        fvs (FileVideoStream): Real time video stream from the specific camera
    """

    def __init__(self, args):
        """Instantiating an instance of CaptureData

        Args:
            args (Object): command line arguments
        """
        self.args = args
        self.run_type = self.args.run_type
        self.camera_position = self.args.camera_position
        self.joystick = None
        # video stream is from standard input from the netcat command
        self.fvs = FileVideoStream('/dev/stdin').start()

        # run a specific method depending on the method name
        getattr(self, self.run_type)()

    def debug(self):
        """Debug mode

        Constantly loop, reading images from the video stream and showing it in a window. Used for adjusting the
        angle of tilt in the cameras and also lining them up horizontally

        Returns:
            None
        """
        while True:
            # get a frame from the video stream
            frame = self.fvs.read()

            # resize the frame
            frame = cv2.resize(frame, (300, 200))

            # flip image horizontally and vertically
            frame = cv2.flip(frame, -1)

            # show the image in a window
            cv2.imshow(self.camera_position, frame)
            if cv2.waitKey(22) & 0xFF == ord('q'):
                break

    def capture(self):
        """Capture mode

        Setup the joystick connection and constantly save the angles and images from the stream in a driving log
        until finished or paused.

        Returns:
            None
        """
        # create the joystick connection
        self.joystick = WheelchairJoystick()

        # whether capturing has started or not
        started = False

        # for recording the images
        image_count = 0

        # where the data is saved to
        output_directory = self.args.output_directory

        # append a data directory to the output directory if it doesn't already exist
        if not output_directory.endswith('data'):
            output_directory = os.path.join(output_directory, 'data')

        # create the images path and driving log path within the output directory
        images_output_directory = os.path.join(output_directory, 'images')
        log_output = os.path.join(output_directory, '{}_driving_log.csv'.format(self.camera_position))

        while True:
            # constantly read frames from the video stream
            frame = self.fvs.read()

            # if the pygame QUIT has been selected, exit the script
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.exit()
                    exit(0)

            # if the capturing process has started
            if started:

                # get inputs controls from the PS3 controller
                angle, speed = self.joystick.get_controls(max_speed=10)

                # if stopped has been pressed on the controller, stop the stream and exit
                if self.joystick.stop_pressed():
                    self.fvs.stop()
                    print('Exiting...')
                    break
                elif self.joystick.pause_pressed():
                    # if pause has been pressed on the controller

                    # short delay and pause the video stream
                    time.sleep(0.5)
                    self.fvs.pause()

                    # constantly loop until unpause is pressed
                    while True:
                        print('Paused...')

                        # keep checking for pygame events - needed for updating controller inputs
                        for event in pygame.event.get():
                            if event.type == pygame.QUIT:
                                pygame.exit()
                                exit(0)

                        # update the joystick inputs
                        self.joystick.update_inputs()

                        # keep checking if unpause has been pressed
                        if self.joystick.pause_pressed():
                            print('Unpaused...')
                            time.sleep(0.5)

                            # unpause pressed, clear and unpause the video stream, break
                            self.fvs.clear()
                            self.fvs.unpause()

                            break

                print('Steering angle: {}'.format(angle))

                # create the image file name, and the absolute path for saving to disk and relative paths for saving
                # it to a row in the driving log .csv file
                image_filename = IMAGE_FILENAME.format(self.camera_position, image_count)
                abs_image_path = os.path.join(images_output_directory, image_filename)
                rel_image_path = os.path.join('images', image_filename)

                # flip horizontally and vertically
                frame = cv2.flip(frame, -1)

                # write images to disk using absolute file name
                cv2.imwrite(abs_image_path, frame)

                # append relative image path and angle results to CSV
                with open(log_output, 'a') as f:
                    f.write('{0},{1:.3f}\n'.format(rel_image_path, math.radians(angle)))

                image_count += 1

            else:
                # the capturing process has not started, check again if the process has started
                started = self.joystick.started_controlling()

                # if it has been started now
                if started:
                    print('Received user input...starting')

                    # only process with the center camera creates the necessary directories
                    if self.camera_position == 'Center':
                        # remove the output directory if it exists and recreate it
                        if os.path.exists(output_directory):
                            shutil.rmtree(output_directory)

                        os.makedirs(images_output_directory)
                    else:
                        # other camera processes have a short time delay
                        time.sleep(0.5)

                    # all processes create their driving logs in the output directory and write the .csv header
                    with open(log_output, 'w') as f:
                        f.write('{},Angle\n'.format(self.camera_position))
                else:
                    # still hasn't started
                    print('Waiting for user input...')


def main(args):
    """Create capture data object using the command line arguments

    Args:
        args (Object): command line arguments

    Returns:
        None
    """
    # start the capturing process if the correct run-type has been given
    if args.run_type:
        CaptureData(args)
        cv2.destroyAllWindows()
    else:
        print('Please enter a run type: \'debug\' or \'capture\'')


if __name__ == '__main__':
    """Entry point for starting the script
    
    Parses the command line arguments and calls the main function
    
    Example Usage:
    nc -l -p 5000 | python capture_data.py <output_directory> Center capture

    nc -l -p 5000 | python capture_data.py <output_directory> Center debug
    
    where nc is the netcat program for streaming video from a camera to the python script
    """
    parser = argparse.ArgumentParser('Capture data from the raspberry pi video streams along with PS3 controller '
                                     'inputs')
    parser.add_argument('output_directory', type=str,
                        help='Path of the output directory of the files (MAKE SURE IT ENDS WITH TRAINING)')
    parser.add_argument('camera_position', type=str, help='Position of the camera (Center, Left or Right)')

    subparsers = parser.add_subparsers(dest='run_type', help='sub-command help')
    parser_debug = subparsers.add_parser('debug', help='Debug mode')
    parser_capture = subparsers.add_parser('capture', help='Capture mode')

    args = parser.parse_args()

    main(args)
