"""
    Filename: utils/concatenate_video.py
    Description: Contains functionality for showing the video footage of the captured data from the 3 cameras
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import argparse
import cv2
import glob
import numpy as np
import time

# global constants
DEBUG = False


def main(args):
    """Main function that runs the Python script

    Shows the 3 camera videos simultaneously

    Args:
        args: (Object) command line arguments

    Returns:
        None
    """

    # append all the video capture footage objects to a list
    vcs = []
    for video_path in glob.glob(args.video_path):
        vcs.append(cv2.VideoCapture(video_path))

    print(vcs)

    while True:
        # continuously loop, grabbing image frames from the videos and stacking them together
        # show the stacked image in a window
        combined_frame = []

        # for each video capture object
        for vc in vcs:

            # read the frames
            success, frame = vc.read()

            # if successful resize the frame and append to the list of frames to stack
            # break from the loop if not successful read
            if success:
                frame = cv2.resize(frame, (200, 200))
                combined_frame.append(frame)
            else:
                break

        # concatenate the frames horizontally to the 1 image
        stack = np.concatenate(tuple(combined_frame), axis=1)

        # show the stacked images
        cv2.imshow('Image', stack)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # add a short delay
        time.sleep(0.2)


if __name__ == '__main__':
    """Main starting point for the Python script, parses the command line arguments and calls the main method
    
    Example Usage: 
    python concatenate_video.py generic-video-path-*.mp4 --debug=True
    """
    parser = argparse.ArgumentParser(description='Combine separate videos for debugging purposes')
    parser.add_argument('video_path', help='Glob path of the videos', type=str)
    parser.add_argument('--debug', help='Debug mode.', type=bool,
                        default=DEBUG)

    args = parser.parse_args()

    main(args)
