"""
    Filename: utils/combine_data.py
    Description: Contains functionality for showing combining the 3 driving logs into the 1 driving log
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import argparse
import os
import pandas as pd


def main(args):
    """Main function that runs the Python script

    Args:
        args: (Object) command line arguments

    Returns:
        None
    """
    data_directory = args.data_directory

    # create a list of dataframes for each of the 3 driving logs by reading their .csv files
    logs = ['Center', 'Left', 'Right']
    dfs = [pd.read_csv(os.path.join(data_directory, '{}_driving_log.csv'.format(log))) for log in logs]

    # create a dictionary containing the number of rows in each driving log file
    row_counts = {}
    for i, df in enumerate(dfs):
        row_counts[logs[i]] = df.shape[0]

    # print the number of rows in each driving log file
    print('Log files', logs)
    print('Row counts', row_counts)

    # get the minimum number of rows from the driving logs
    min_rows = row_counts[min(row_counts, key=lambda x: row_counts.get(x))]
    print('Minimum rows', min_rows)

    # create the dataframe to hold the entire driving log data
    combined_df = pd.DataFrame(columns=['Center', 'Left', 'Right', 'Angle'])

    # use the minimum rows as the iteration index ensuring each driving log has the particular row
    for i in range(min_rows):

        # grab the angles from each of the driving logs at the particular row
        angles = [df.iloc[i]['Angle'] for df in dfs]

        # calculate the average angle between the driving logs for that row
        average_angle = round(sum(angles) / 3, 3)

        # add a row to the concatenated driving log
        # each row contains the path to the centre, left and right images
        # as well as the averaged steering angle
        combined_df.loc[i] = [
            dfs[0].iloc[i]['Center'],
            dfs[1].iloc[i]['Left'],
            dfs[2].iloc[i]['Right'],
            average_angle
        ]

    # write the final driving log to a .csv file
    combined_df.to_csv(os.path.join(data_directory, 'driving_log.csv'), index=False)


if __name__ == '__main__':
    """Main starting point for the Python script, parses the command line arguments and calls the main method
    
    Example Usage: 
    python combine_data.py <data_directory>
    """
    parser = argparse.ArgumentParser(description='Combine separate video and angle data into the one dataset')
    parser.add_argument('data_directory', type=str, help='Directory of the data directory containing the separate data '
                                                         'to be combined.')

    args = parser.parse_args()

    main(args)
