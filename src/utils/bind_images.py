"""
    Filename: utils/bind_images.py
    Description: Contains functionality for combining images to a video
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import argparse
import cv2
import glob
import os
import re

# global constants
DEBUG = False
WIDTH, HEIGHT = (1280, 720)


def natural_sort(l):
    """Method to sort a list based on increasing numerical order

    Args:
        l (List): List of filename entries to sort in order

    Returns:
        (List): List sorted in natural order
    """
    # convert the text to an integer if it's a digit, else change it to lowercase
    convert = lambda text: int(text) if text.isdigit() else text.lower()

    # run regex on each entry of the list looking for the digits in their name and convert them to integers
    # for ordering based on their number entry
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    # return the sorted list using the lambda function as the key to sort by
    return sorted(l, key=alphanum_key)


def main(args):
    """Main method for binding images to a video

    Args:
        args (Object): command line arguments

    Returns:
        None
    """
    # create an instance of the video writer that creates a video called "output.avi" saved to the output directory
    # given in the command line
    data_directory = args.data_directory
    writer = cv2.VideoWriter(os.path.join(data_directory, 'output.avi'), cv2.VideoWriter_fourcc(*'MJPG'),
                             30, (WIDTH, HEIGHT))

    # loop over the images within the glob in their natural sort order
    for filename in natural_sort(glob.glob(data_directory + args.glob)):
        # read the image
        image = cv2.imread(filename)

        # if in debug mode, show the image in a window
        if args.debug:
            cv2.imshow('Image', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        # resize to the specified size and write the image to the video writer
        image = cv2.resize(image, (WIDTH, HEIGHT))
        writer.write(image)

    # release the video writer and destroy all OpenCV windows
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    """Main starting point for the Python script

    Calls the main method with the command line arguments
    
    Example Usage: 
    python bind_images.py <data_directory> generic-image-number-*.jpg --width=1280 --height=720 --debug=True
    """
    parser = argparse.ArgumentParser(description='Create video from directory of images.')
    parser.add_argument('data_directory', help='Image directory.', type=str)
    parser.add_argument('glob', help='Generic image name.', type=str)
    parser.add_argument('--width', help='Width of resize.', type=int, default=WIDTH)
    parser.add_argument('--height', help='Height of resize.', type=int, default=HEIGHT)
    parser.add_argument('--debug', help='Debug mode.', type=bool, default=DEBUG)

    args = parser.parse_args()

    main(args)
