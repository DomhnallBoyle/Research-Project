"""
    Filename: utils/android_app.py
    Description: Contains functionality for creating the driving log from the recorded data from the android app
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import argparse
import cv2
import os
import pandas as pd
import shutil
import time

# global constants
DEBUG = False
CSV_FILE = 'AVDataCapture_data.csv'
VIDEO_FILE = 'AVDataCapture_video.mp4'
WORKING_DIR = os.path.abspath(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(WORKING_DIR, 'results')
IMAGES_DIR = os.path.join(RESULTS_DIR, 'images')
OUTPUT_CSV_FILE = os.path.join(RESULTS_DIR, 'results.csv')
WIDTH, HEIGHT = (960, 540)


def main(args):
    """Main method for converting the android data to driving logs for training

    Args:
        args (Object): command line arguments

    Returns:
        None
    """
    data_directory = args.data_directory

    # remove the output directory if it exists
    if os.path.exists(RESULTS_DIR):
        shutil.rmtree(RESULTS_DIR)

    # recreate the results directory
    os.makedirs(IMAGES_DIR)

    # instance of the video capture pointed to the video footage
    video_capture = cv2.VideoCapture(os.path.join(data_directory, VIDEO_FILE))

    # record the image names from the video footage
    image_names = []

    # record the number of images
    count = 1

    # while there are still frames from the video
    while video_capture.isOpened():

        # read a frame from the footage
        success, frame = video_capture.read()

        # if successful read
        if success:

            # resize the frame
            frame = cv2.resize(frame, (args.width, args.height))

            # debug mode - show the frame in a window
            if args.debug:
                cv2.imshow('Frame', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # create the image name
            image_name = os.path.join(IMAGES_DIR, '{}.jpg'.format(count))

            # append to the list
            image_names.append(image_name)

            # save the image to disk using this absolute path
            cv2.imwrite(image_name, frame)

            # increment the image count
            count += 1
        else:
            break

    # release the video capture footage when finished
    video_capture.release()

    # read the .csv file containing the steering angles
    df = pd.read_csv(os.path.join(data_directory, CSV_FILE))
    num_rows = df.shape[0]

    # print the number of angles and images
    print('Number of Angles: {}'.format(num_rows))
    print('Number of Images: {}'.format(len(image_names)))

    # if there are more rows than images, reduce the number of rows to the number of images available
    # if more images, reduce the number of images to the number of rows available
    if num_rows > len(image_names):
        df = df[:-(num_rows - len(image_names))]
    elif num_rows < len(image_names):
        image_names = image_names[:num_rows]

    # apply a function to the angle column, converting it to 2 decimal places
    df['Angle'] = df['Angle'].apply(lambda x: '{0:.2f}'.format(x))

    # create a new column with the image names
    df['Image'] = image_names

    # save the updated .csv file
    df.to_csv(OUTPUT_CSV_FILE, encoding='utf8', index=False)

    # debugging
    # iterate through the number of rows in the .csv file
    for index, row in df.iterrows():

        # extract the angle and image name from each row
        angle = row['Angle']
        image_name = row['Image']

        # read the image in colour
        image = cv2.imread(image_name, 1)

        # add text to the image showing the angle at that point
        cv2.putText(image, '{}'.format(angle), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)

        # show the image with the steering angle overlayed
        cv2.imshow('Frame', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

        # time delay in debug mode
        if args.debug:
            time.sleep(0.1)

    # finished, destroy all open image windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    """Main starting point for the Python script

    Calls the main method with the command line arguments
    
    Example Usage: 
    python android_app.py <data_directory> --width=1280 --height=720 --debug=True
    """
    parser = argparse.ArgumentParser(description='Pre-processing data captured by the AVDataCapture Application.')
    parser.add_argument('data_directory', help='Directory containing the AVDataCapture data.')
    parser.add_argument('--width', help='Width of output images.', type=int, default=WIDTH)
    parser.add_argument('--height', help='Height of output images.', type=int, default=HEIGHT)
    parser.add_argument('--debug', help='Debug mode.', type=bool, default=DEBUG)

    args = parser.parse_args()

    main(args)
