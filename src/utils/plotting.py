"""
    Filename: utils/plotting.py
    Description: Contains functionality for plotting graphs of the CNN training process
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import argparse
import json
import os
import matplotlib.pyplot as plt

# global constants
# dictionary containing the loss functions for regression and classification
LOSSES = {
    'regression': 'MSE',
    'classification': 'Categorical Cross-Entropy'
}


def plot_line_graph(x_points, y_points, title, x_label, y_label, legend=None, colours=['blue', 'red'], x_markers={},
                    save_path=None):
    """Function to plot a line graph

    Args:
        x_points (List): 2d list containing the x points to be plotted on the graphs
        y_points (List): 2d list containing the associated y points to be plotted on the graph
        title (String): title of the graph to be plotted
        x_label (String): x axis label of the graph
        y_label (String): y axis label of the graph
        legend (List): list of legend strings for the plotted lines
        colours (List): colours for the lines (len(x_points) == len(y_points) == len(colours))
        x_markers (Dictionary): contain the positions for vertical markers to be placed on points of the x-axis
        save_path (String): save path for saving the image to disk

    Returns:
        None
    """
    # ensure same number of dimensions (number of plotted lines == len(x_points) == len(y_points)
    assert(len(x_points) == len(y_points))

    # enumerate through each pair x and y coordinate lists in the 2d lists
    for i, (x, y) in enumerate(zip(x_points, y_points)):
        # plot the coordinate lists and the corresponding colours
        plt.plot(x, y, color=colours[i])

    # add the title, x labels and y labels to the graph
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # loop through the vertical x-axis markers in the dictionary
    # adding them to the plot with their labels as the values
    for k, v in x_markers.items():
        plt.axvline(x=v)
        plt.text(v, 0.5, k, rotation=90, verticalalignment='center')

    # if the legend strings are given, add them to the plot
    if legend:
        plt.legend(legend, loc='upper right')

    # save the graph if a path is given
    if save_path:
        plt.savefig(save_path)

    # show the plot
    plt.show()


def plot_history_graphs(d, output_dir, type):
    """Wrapper around the plot_line_graph function to plot the history graphs of the training process

    Args:
        d (Dictionary): containing the history of the training process
        output_dir (String): the output directory to save the plot to
        type (String): type of problem the network solves i.e. regression/classification

    Returns:
        None
    """
    # depending on the type of problem, construct a dictionary containing the metric to be plotted as the key
    # and the list of strings to extract the relevant training and validation metric values
    if type == 'regression':
        history_keys = {
            'loss (MSE)': ['loss', 'val_loss'],
            'R^2': ['r_square', 'val_r_square'],
            'RMSE': ['rmse', 'val_rmse'],
            'MAE': ['mean_absolute_error', 'val_mean_absolute_error'],
            'MAPE': ['mean_absolute_percentage_error', 'val_mean_absolute_percentage_error']
        }
    else:
        history_keys = {
            'loss (Categorical Cross-Entropy)': ['loss', 'val_loss'],
            'Accuracy': ['acc', 'val_acc']
        }

    # the x points should just be the increasing number of epochs - use the length of the loss values
    # in the dictionary to construct this
    x_points = [i + 1 for i in range(len(d['loss']))]

    # for every metric in the dictionary
    for key in history_keys.keys():
        # plot the training and validation line graphs using the metric
        # extract the appropriate training/validation metric from the history dictionary to be used as the y points
        plot_line_graph(x_points=[x_points, x_points],
                        y_points=[d[history_keys[key][0]], d[history_keys[key][1]]],
                        title='Model {} over all epochs'.format(key),
                        x_label='Epoch',
                        y_label=key,
                        legend=['Training', 'Validation'],
                        save_path=os.path.join(output_dir, '{}.png'.format(key)))


def plot_first_epoch_graphs(d, output_dir, loss):
    """Function to the training and validation loss histories during the first epoch

    Args:
        d (Dictionary): containing the history of the training and validation losses during the first epoch
        output_dir (String): path to save the plots to
        loss (String): loss metric that is used in the training

    Returns:
        None
    """
    # extract the training and validation losses from the first epoch from the graphs
    train_losses = d['training_losses']
    val_losses = d['validation_losses']

    # x points for the graphs, they're just the number of batches used in the training and validation
    x_points = [[i + 1 for i in range(len(train_losses))],
                [i + 1 for i in range(len(val_losses))]]

    # y points are just the training and validation losses
    y_points = [train_losses, val_losses]

    # plot the training loss graph during the first epoch
    plot_line_graph(x_points=[x_points[0]], y_points=[y_points[0]],
                    title='Model training loss ({}) during the first epoch'.format(loss),
                    x_label='Batch Number',
                    y_label='Loss',
                    save_path=os.path.join(output_dir, 'first_epoch_loss_training.png'))

    # plot the validation loss graph during the first epoch
    plot_line_graph(x_points=[x_points[1]], y_points=[y_points[1]],
                    title='Model validation loss ({}) during the first epoch'.format(loss),
                    x_label='Batch Number',
                    y_label='Loss',
                    save_path=os.path.join(output_dir, 'first_epoch_loss_validation.png'))


def plot_graphs(history_d, first_epoch_d, output_dir, type):
    """Function wrapper around the plot_history_graphs and plot_first_epoch_graphs function

    This function is used at the end of the training process

    Args:
        history_d (Dictionary): containing the history of the training process
        first_epoch_d (Dictionary): containing the history of the first epoch losses
        output_dir (String): path to save the plots to
        type (String): type of training i.e. regression/classification

    Returns:
        None
    """
    plot_history_graphs(history_d, output_dir, type)
    plot_first_epoch_graphs(first_epoch_d, output_dir, LOSSES[type])


def main(args):
    """Function that operates from the command line that plots the graphs from the saved JSON containing the training
    history

    Args:
        args (Object): containing the command line arguments

    Raises:
        FileNotFoundError: if the JSON file doesn't exist

    Returns:
        None
    """
    # create the paths to the JSON files
    history = os.path.join(args.output_dir, 'history.json')
    first_epoch_losses = os.path.join(args.output_dir, 'first_epoch_losses.json')

    # try opening the full training history file
    # raise and print the exception if the file is not found
    try:
        with open(history) as f:
            # load the JSON
            history_d = json.load(f)

            # plot and save the graphs
            plot_history_graphs(history_d, args.output_dir, args.type)
    except FileNotFoundError as e:
        print(e)
        pass

    # try opening the first epoch history file
    # raise and print the exception if the file is not found
    try:
        with open(first_epoch_losses) as f:
            # load the JSON
            first_epoch_d = json.load(f)

            # plot and save the graphs
            plot_first_epoch_graphs(first_epoch_d, args.output_dir, LOSSES[args.type])
    except FileNotFoundError as e:
        print(e)
        pass


if __name__ == '__main__':
    """Main starting point for the Python script, parses the command line arguments and calls the main method
    
    Example Usage: 
    python plotting.py <output_dir> --type=regression
    """
    parser = argparse.ArgumentParser('For plotting line graphs from training')
    parser.add_argument('output_dir', type=str, help='Output directory to save the graphs to')
    parser.add_argument('--type', type=str, default='regression')

    args = parser.parse_args()

    main(args)
