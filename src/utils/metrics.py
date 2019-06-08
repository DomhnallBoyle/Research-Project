"""
    Filename: utils/metrics.py
    Description: Contains functions for calculating training metrics
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import numpy as np


def rmse(predictions, groundtruth):
    """Function for calculating the Root Mean Squared Error metric

    The RMSE is the square root of the average of squared differences (MSE)

    Args:
        predictions (List): list of predicted angles
        groundtruth (List): list of groundtruth angles

    Returns:
        (Float): the calculated RMSE
    """
    return np.sqrt(np.sum(np.square(groundtruth - predictions)) / len(groundtruth))


def mae(predictions, groundtruth):
    """Function for calculating the Mean Absolute Error metric

    The MAE is the average of absolute differences between the prediction and groundtruth values

    Args:
        predictions (List): list of predicted angles
        groundtruth (List): list of groundtruth angles

    Returns:
        (Float): the calculated MAE
    """
    return np.sum(np.absolute(groundtruth - predictions)) / len(groundtruth)


def r_squared(predictions, groundtruth):
    """Function for calculating the R^2 error metric

    The R^2 is the co-efficient of determination

    Args:
        predictions (List): list of predicted angles
        groundtruth (List): list of groundtruth angles

    Returns:
        (Float): the calculated R^2
    """
    # find the
    rss = np.sum(np.square(groundtruth - predictions))
    tss = np.sum(np.square(groundtruth - np.mean(groundtruth)))

    return 1 - (rss / tss)
