"""
    Filename: utils/check_data.py
    Description: Contains functionality for checking the validity of the captured data
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import argparse
import cv2
import math
import numpy as np
import os
import pandas as pd
from PIL import Image

# local source imports
from drawing import get_line_points

# global constants
WIDTH = 500
HEIGHT = 300
RADIANS_90 = math.radians(90)


def main(args):
    """Main function that runs the Python script

    Validates the captured data from the 3 driving logs before combining using combine_data.py

    Args:
        args: (Object) command line arguments

    Raises:
        IndexError: if the row index doesn't exist in any of the pandas dataframes
        OSError: if a specific image file is corrupt

    Returns:
        None
    """
    data_directory = args.data_directory

    # read the 3 captured driving logs
    logs = ['Center', 'Left', 'Right']
    dfs = [pd.read_csv(os.path.join(data_directory, '{}_driving_log.csv'.format(log))) for log in logs]

    # get the row counts for each of the driving logs
    row_counts = {}
    for i, df in enumerate(dfs):
        row_counts[logs[i]] = df.shape[0]

    print('Log files', logs)
    print('Row counts', row_counts)

    # find the minimum number of rows in the dataframes
    min_rows = row_counts[min(row_counts, key=lambda x: row_counts.get(x))]
    print('Minimum rows', min_rows)

    # make sure that there are entries in the driving logs
    if min_rows != 0:

        # to record row indexes to drop that are invalid
        indexes_to_drop = []

        # for each row index
        for i in range(min_rows):
            try:
                # get the Centre, left and right image paths from the specific driving logs
                image_paths = [dfs[0].iloc[i]['Center'], dfs[1].iloc[i]['Left'], dfs[2].iloc[i]['Right']]

                valid = True

                # for each image path
                for image_path in image_paths:

                    # check that the image path is actually a file
                    full_image_path = os.path.join(data_directory, image_path)
                    if not os.path.isfile(full_image_path):
                        # if not, print the image path and add it to the indexes to drop
                        # break from the current row
                        print(image_path, 'does not exist')
                        valid = False
                        indexes_to_drop.append(i)
                        break

                    # check the validity of the image file itself - that it is an actual image
                    # and not a corrupt file
                    try:
                        image = Image.open(full_image_path)
                        image.verify()
                    except OSError as e:
                        # corrupt file; add the row index to the indexes to drop
                        # break from the current row
                        print(e)
                        valid = False
                        indexes_to_drop.append(i)
                        break

                # if the images are valid
                if valid:
                    # get the angles from each of the driving logs and average them
                    angles = [dfs[j].iloc[i]['Angle'] for j in range(3)]
                    mean_angle = np.mean(angles)

                    # read the images and resize them to the correct width and height
                    images = [cv2.imread(os.path.join(data_directory, image_path)) for image_path in image_paths]
                    images = [cv2.resize(image, (WIDTH, HEIGHT)) for image in images]

                    # for visualisation purposes only, concatenate the images and add text to the combined image
                    combined_image = np.concatenate((images[1], images[0], images[2]), axis=1)
                    cv2.putText(combined_image,
                                'Image: {}'.format(i),
                                (10, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2, (0, 0, 255))
                    cv2.putText(combined_image,
                                'Radians: {}'.format(round(mean_angle, 3)),
                                (10, 150),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2, (0, 0, 255))
                    cv2.putText(combined_image,
                                'Degrees: {}'.format(round(math.degrees(mean_angle), 3)),
                                (10, 250),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                2, (0, 0, 255))

                    # get the line points for the groundtruth line and add it to the image
                    line_points = get_line_points(WIDTH*3, HEIGHT, -RADIANS_90 - mean_angle)
                    cv2.line(combined_image, line_points[0], line_points[1], (0, 255, 0), 2)

                    # show the image
                    cv2.imshow('Frame', combined_image)
                    if cv2.waitKey(10) & 0xFF == ord('q'):
                        break

            except IndexError as e:
                # row does not exist, print information and continue
                print('Index Error at {}'.format(i))
                print(e)
                pass

        # update the 3 driving logs by removing the indexes of the rows that
        # contained invalid images
        for i, df in enumerate(dfs):
            df.drop(indexes_to_drop, inplace=True)
            df.to_csv(os.path.join(data_directory, '{}_driving_log_new.csv'.format(logs[i])))

    else:
        print('No CSV data has been recorded')

    cv2.destroyAllWindows()


if __name__ == '__main__':
    """Main starting point for the Python script, parses the command line arguments and calls the main method
    
    Example Usage: 
    python check_data.py <data_directory>
    """
    parser = argparse.ArgumentParser(description='Check the images exist within the CSV files before combining the data')
    parser.add_argument('data_directory', type=str, help='Absolute directory path of the data directory containing the '
                                                         'separate data to be checked.')

    args = parser.parse_args()

    main(args)
