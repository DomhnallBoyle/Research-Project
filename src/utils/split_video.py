"""
    Filename: utils/split_video.py
    Description: Contains functionality for splitting a video into separate image frames
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import argparse
import cv2
import os

# global constants
WIDTH, HEIGHT = (960, 540)
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__)) + '/'
IMAGE_DIR = CURRENT_DIR + 'images/'


def main(args):
    """Main function for running the Python script

    Args:
        args: (Object) command line arguments

    Returns:
        None
    """
    # create the video capture instance with the path of the video
    video_capture = cv2.VideoCapture(args.video)
    count = 1

    # make the output directory for the images if it doesn't already exist
    if not os.path.exists(IMAGE_DIR):
        os.mkdir(IMAGE_DIR)

    # while there are still images in the video to be read
    while video_capture.isOpened():

        # read an image from the video capture object
        success, frame = video_capture.read()

        # if the read of the frame was successful
        if success:

            # resize it
            frame = cv2.resize(frame, (args.width, args.height))

            # if in debug mode show the frame just read
            if args.debug:
                cv2.imshow('Frame', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # write the image to disk
            image_name = IMAGE_DIR + '{}.jpg'.format(count)
            cv2.imwrite(image_name, frame)

            count += 1
        else:
            break

    # destroy all the open image windows and release the video capture
    cv2.destroyAllWindows()
    video_capture.release()


if __name__ == '__main__':
    """Main starting point for the Python script, parses the command line arguments and calls the main method
    
    Example Usage: 
    python split_video.py <video_path> --width=1280 --height=720 --debug=True
    """
    parser = argparse.ArgumentParser(description='Converts a video to '
                                                 'individual image files.')
    parser.add_argument('video', help='Absolute directory of video file.')
    parser.add_argument('--width', help='Width of output images.', type=int,
                        default=WIDTH)
    parser.add_argument('--height', help='Height of output images.', type=int,
                        default=HEIGHT)
    parser.add_argument('--debug', help='Debug mode.')
    args = parser.parse_args()

    main(args)
