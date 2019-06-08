"""
    Filename: models/base_model.py
    Description: Contains functionality for training, testing and evaluating of models that derive this class
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import argparse
import cv2
import json
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random as rn
from abc import ABC, abstractmethod
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
from vis.visualization import visualize_saliency, visualize_cam, overlay
from vis.utils import utils
import matplotlib.cm as cm

# local source imports
from utils import BatchCallback, DLBot, draw_on_frame, get_line_points, plot_graphs, plot_line_graph, TelegramBotCallback
from utils import metrics

# global constants
LEARNING_RATE = 1e-4
STEPS_PER_EPOCH = 20000
BATCH_SIZE = 40
EPOCHS = 10
TRAINING_TEST_SPLIT = 0.2
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', None) # environment variable (privacy)

# setting the seed environment variable
os.environ['PYTHONHASHSEED'] = '0'

# Setting the seed for numpy-generated random numbers
np.random.seed(2019)

# Setting the seed for python random numbers
# used to get the same training process each time if doing reruns
rn.seed(2019)

# global variables
validation_combinations = []


def rmse(y_true, y_pred):
    """Calculate the RMSE metric for keras models during training

    Args:
        y_true (List): list of groundtruth steering angles
        y_pred (List): list of predicted steering angles

    Returns:
        (Float): the RMSE
    """
    # root mean squared error (rmse) for regression
    # axis=-1
    # print(K.int_shape(y_pred))
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=0))


def r_square(y_true, y_pred):
    """Calculate the R^2 metric for keras models during training

    Coefficient of determination (R^2) for regression

    Args:
        y_true (List): list of groundtruth steering angles
        y_pred (List): list of predicted steering angles

    Returns:
        (Float) the co-efficient of determination
    """
    # calcuate the sum of squared residuals
    SS_res = K.sum(K.square(y_true - y_pred))

    # calculate the total sum of squares
    SS_tot = K.sum(K.square(y_true - K.mean(y_true)))

    return 1 - SS_res / (SS_tot + K.epsilon())


def string_list(s):
    """Used for parsing the list of training log variations in the CLI

    Args:
        s (String): string containing a list of driving log paths

    Returns:
        (List): converted to a list of strings using JSON
    """
    return json.loads(s)


class BaseModel(ABC):
    """Abstract base class giving generic training, testing and evaluation functionality to the CNN subclasses

    Attributes:
        name (String): name/short description of the model
        channels (Integer): number of channels in the image (usually 3)
        height (Integer): height of the images used in the model
        width (Integer): width of the images used in the model
        type (String): type of problem the model tries to solve i.e. classification/regression
        parser (ArgumentParser): for parsing the command line arguments given through the CLI
        model (SequentialModel): keras model to be built or loaded from disk
        batch_callback (BatchCallback): custom callback used to perform additional training tasks
    """

    def __init__(self, name, size, type):
        """Abstract base class, cannot instantiate

        Args:
            name (String): name/short description of the model
            size (Tuple): contains the number of channels, height and width of the images
            type (String): type of problem the network tries to solve i.e. classification/regression
        """
        self.name = name
        self.channels, self.height, self.width = size
        self.type = type
        self.parser = self.build_parser()
        self.args = self.parser.parse_args()
        self.model = None
        self.batch_callback = BatchCallback(save_directory=self.args.output_dir)

        print(self.args.__dict__)

        # run the specific function based on the CLI argument
        if self.args.run_type in ['training', 'testing', 'evaluate', 'debug']:
            getattr(self, self.args.run_type)()
        else:
            # print help if function name doesn't exist
            self.parser.print_help()

    def build_parser(self):
        """Build the CLI argument parser

        4 sub-parsers; training, testing, evaluate and debug

        Returns:
            (ArgumentParser): to parse the arguments given through the CLI
        """
        parser = argparse.ArgumentParser(self.name)

        subparsers = parser.add_subparsers(dest='run_type', help='sub-command help')

        parser_training = subparsers.add_parser('training', help='Training help')
        parser_training.add_argument('csv_paths', help='List of csv paths for training e.g. straights and corners', type=string_list)
        parser_training.add_argument('output_dir', help='For outputting the weights and logs of the training')
        parser_training.add_argument('--model_path', help='Path of the model to keep training', type=str, default=None)
        parser_training.add_argument('--crop_to', help='Area of the image to crop', type=int, default=0)
        parser_training.add_argument('--split_test_size', help='Size of training-test split', default=TRAINING_TEST_SPLIT)
        parser_training.add_argument('--batch_size', help='Batch size', type=int, default=BATCH_SIZE)
        parser_training.add_argument('--steps_per_epoch', help='Samples per epoch', type=int, default=STEPS_PER_EPOCH)  # number of iterations?
        parser_training.add_argument('--epochs', help='Number of epochs', type=int, default=EPOCHS)
        parser_training.add_argument('--learning_rate', help='Learning rate', default=LEARNING_RATE, type=float)

        parser_testing = subparsers.add_parser('testing', help='Testing help')
        parser_testing.add_argument('model_path', help='Path of trained model')
        parser_testing.add_argument('groundtruth_path', help='Groundtruth angles in a CSV file')
        parser_testing.add_argument('output_dir', help='Directory to save video/image prediction')
        parser_testing.add_argument('--video_path', help='Path of video to test')
        parser_testing.add_argument('--image_path', help='Path of the image to test')
        parser_testing.add_argument('--crop_to', help='Area of the image to crop', type=int, default=0)

        parser_evaluate = subparsers.add_parser('evaluate', help='Evaluate help')
        parser_evaluate.add_argument('model_path', help='Path of trained model')
        parser_evaluate.add_argument('csv_paths', help='List of csv paths for training e.g. straights and corners', type=string_list)
        parser_evaluate.add_argument('output_dir', help='Directory to save results to')
        parser_evaluate.add_argument('--crop_to', help='Aread of the iamge to crop', type=int, default=0)
        parser_evaluate.add_argument('--split_test_size', help='Size of training-test split', default=TRAINING_TEST_SPLIT)

        parser_debug = subparsers.add_parser('debug', help='Debug help')
        parser_debug.add_argument('csv_paths', help='List of csv paths e.g. straights and corners', type=string_list)
        parser_debug.add_argument('output_dir', help='For outputting debug graphs and images')
        parser_debug.add_argument('--image_path', help='Image to show neural network debug information on', type=str)
        parser_debug.add_argument('--crop_to', help='Area of the image to crop', type=int, default=0)
        parser_debug.add_argument('--model_path', help='Path of trained model')
        parser_debug.add_argument('--split_test_size', help='Size of training-test split', default=TRAINING_TEST_SPLIT)

        return parser

    def get_generic_callbacks(self):
        """Create the generic callbacks

        Creates generic callbacks for the training process e.g. telegram, batch_callback and model_checkpoint
        The callbacks supply methods to be ran before/after epochs, before/after batches etc

        Returns:
            None
        """
        callbacks = []

        # if the telegram token exists, use the telegram callback
        if TELEGRAM_TOKEN:
            bot = DLBot(token=TELEGRAM_TOKEN, user_id=None)
            telegram_callback = TelegramBotCallback(bot)
            callbacks.append(telegram_callback)
        else:
            # no telegram token environment variable
            print('No Telegram Token - not using Telegram callback')

        # callback to save the model after every 5 epochs
        checkpoint_path = os.path.join(self.args.output_dir, 'model-{epoch:03d}.h5')

        # save model after every 5 epochs
        checkpoint_callback = ModelCheckpoint(checkpoint_path, save_weights_only=True, period=5)

        # callback to view the training model using TensorBoard
        logs_path = os.path.join(self.args.output_dir, 'logs')
        tensorboard_callback = TensorBoard(logs_path)

        callbacks.extend([self.batch_callback, checkpoint_callback, tensorboard_callback])

        return callbacks

    def load_model(self):
        """Load the model if a model path has been given through the CLI

        Returns:
            None
        """
        if self.args.model_path:
            # requires building the model first - only the model weights are saved to disk
            self.model = self.build_model()
            self.model.load_weights(self.args.model_path)

    def load_image(self, image_path):
        """Load the image given an absolute image path

        Args:
            image_path (String): absolute path of the image

        Returns:
            (List): 3D list representing the read RGB image
        """
        if not os.path.exists(image_path):
            raise OSError('Image not found: ' + image_path)

        return cv2.imread(image_path, 1)

    def load_training_data(self):
        """Load the training data using the driving log variations passed in through the CLI

        Returns:
            (Tuple): containing the separated training and validation images and groundtruth labels
        """
        # use pandas to read each driving log, save to list
        data_combinations = [pd.read_csv(csv_path) for csv_path in self.args.csv_paths]

        # for recording the training and validation data
        training_x, validation_x, training_y, validation_y = [], [], [], []

        # each data combination is split 80/20 so validation gets last 20% of all combinations for testing
        for i, combination in enumerate(data_combinations):
            # get the directory of the driving log
            directory = os.path.dirname(self.args.csv_paths[i])

            # convert the image paths from relative to absolute
            data_combinations[i][['Left', 'Center', 'Right']] = directory + '/' + combination[['Left', 'Center', 'Right']].astype(str)

            # extract the image and groundtruth values
            x = data_combinations[i][['Left', 'Center', 'Right']].values
            y = data_combinations[i]['Angle'].values

            # split the driving log (80/20) for training and testing
            # no shuffling because we want test data to have samples that have been left out of training
            # use the last 20% from each data variation for the validation data
            x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=self.args.split_test_size,
                                                                  random_state=0, shuffle=False)

            # record the point of each validation combination
            if i == 0:
                validation_combinations.append(len(y_valid))
            else:
                validation_combinations.append(validation_combinations[i-1] + len(y_valid))

            # add the split data to the appropriate list
            training_x.extend(x_train)
            validation_x.extend(x_valid)
            training_y.extend(y_train)
            validation_y.extend(y_valid)

        return training_x, validation_x, training_y, validation_y

    def load_groundtruth_data(self, csv_path):
        """Load the groundtruth data given a driving log path

        Args:
            csv_path (String): absolute driving log path

        Returns:
            (List): list of groundtruth angles
        """
        data = pd.read_csv(csv_path)
        y = data['Angle'].values

        return y

    def find_groundtruth_by_image_name(self, csv_path, image_name):
        """Finds a steering angle based on the image angle in a driving log

        Args:
            csv_path (String): absolute driving log path
            image_name (String): name of the image to search for

        Returns:
            (Float): groundtruth steering angle
        """
        # read the csv using pandas
        data = pd.read_csv(csv_path)

        # find the row where the centre, left or right images have a name that is the same as the passed in image name
        # there should only be 1 row extracted
        row = data[(data.Center == image_name) | (data.Left == image_name) | (data.Right == image_name)]

        # return the groundtruth angle from this row
        return row['Angle'].iloc[0]

    def plot_training_results(self, history):
        """Plot the results of the training

        Plot the training and validation history over the number of epochs
        Plot the training and validation losses during the first epoch

        Args:
            history (keras.History): history object of the training process

        Returns:
            None
        """
        # Plot training & validation loss values
        plot_graphs(history_d=history.history, first_epoch_d={
            'training_losses': list(self.batch_callback.first_epoch_training_losses.values()),
            'validation_losses': list(self.batch_callback.first_epoch_val_losses.values())
        }, output_dir=self.args.output_dir, type=self.type)

    def testing(self):
        """Test the trained models

        Test single images or frames of a video on a trained model. Results are saved to disk.

        Args:
            None

        Returns:
            None
        """
        self.load_model()

        if self.args.video_path:
            # open the video if a video path has been given as argument
            video_capture = cv2.VideoCapture(self.args.video_path)

            # open a video writer for the output
            video_out = cv2.VideoWriter(os.path.join(self.args.output_dir, 'predictions.avi'),
                                        cv2.VideoWriter_fourcc(*'MJPG'), 30, (1280, 720))

            # load the groundtruth data
            groundtruth_data = self.load_groundtruth_data(self.args.groundtruth_path)

            index = 0
            while video_capture.isOpened():
                # constantly read frames from the video
                success, frame = video_capture.read()

                if success:
                    # extract the groundtruth data at that particular frame
                    groundtruth = groundtruth_data[index]

                    # pass the frame to the model and get a prediction
                    prediction = self.test_model(frame)[0][0]

                    # draw lines on the frame representing the prediction and groundtruth
                    frame = draw_on_frame(frame, prediction, groundtruth)

                    # write the frame to the video
                    video_out.write(frame)

                    # show the image in a window
                    cv2.imshow('Image', frame)
                    # press Q on keyboard to exit
                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break

                    index += 1
                else:
                    break

            # release the video output and capture when finished
            video_out.release()
            video_capture.release()
        elif self.args.image_path:
            # image path has been given instead to test

            # extract the image name from the image path
            image_name = '/'.join(self.args.image_path.split('/')[-2:])

            # find the groundtruth angle using the image name
            groundtruth = self.find_groundtruth_by_image_name(self.args.groundtruth_path, image_name=image_name)

            # read the image
            image = cv2.imread(self.args.image_path, 1)

            # make the prediction
            prediction = self.test_model(image)[0][0]

            # draw lines on the frame representing the prediction and groundtruth
            frame = draw_on_frame(image, prediction, groundtruth)

            # write the frame to disk
            cv2.imwrite(os.path.join(self.args.output_dir, self.args.image_path.split('/')[-1]), frame)

            # show the image
            cv2.imshow('Image', frame)
            cv2.waitKey(0)

    def evaluate(self):
        """Evaluate the regression model

        Finds the best and worst predicted images as well as draws their saliency and activation maps

        Returns:
            None
        """
        # load the model
        self.load_model()

        # load the training data
        training_x, validation_x, training_y, validation_y = self.load_training_data()

        # storage of the predictions, best and worst predicted images
        predictions = []
        centre_image_paths = []
        best_3_predicted = {}
        worst_3_predicted = {}

        # enumerate through the validation data
        for i, images in enumerate(validation_x):
            # testing the centre image
            centre_image_path = images[1]
            centre_image_paths.append(centre_image_path)
            centre_image = cv2.imread(centre_image_path, 1)

            # get the prediction and append to the list
            prediction = self.test_model(centre_image)[0][0]
            predictions.append(prediction)

        # convert to numpy arrays
        predictions = np.asarray(predictions)
        groundtruth = np.asarray(validation_y)

        # calculate the regression metrics
        print('RMSE', metrics.rmse(predictions, groundtruth))
        print('MAE', metrics.mae(predictions, groundtruth))
        print('R^2', metrics.r_squared(predictions, groundtruth))

        # calculate the absolute differences between prediction and groundtruth lists
        absolute_diffs = np.abs(predictions - groundtruth)

        # get the indexes of the best 3 and worst 3 predicted
        max_indexes = np.argsort(-absolute_diffs)[:3]
        min_indexes = np.argsort(absolute_diffs)[:3]

        # append the best and worst 3 predicted to the dictionary
        # image path is the key, prediction and groundtruth the dictionary value
        for i in range(3):
            best_3_predicted[centre_image_paths[min_indexes[i]]] = {
                'prediction': predictions[min_indexes[i]],
                'groundtruth': validation_y[min_indexes[i]]
            }

            worst_3_predicted[centre_image_paths[max_indexes[i]]] = {
                'prediction': predictions[max_indexes[i]],
                'groundtruth': validation_y[max_indexes[i]]
            }

        print('Best 3 predictions', best_3_predicted)
        print('Worst 3 predictions', worst_3_predicted)

        # show the best and worst 3 predicted images, save them to disk
        self.show_images(best_3_predicted, os.path.join(self.args.output_dir, 'best_predicted.png'))
        self.show_images(worst_3_predicted, os.path.join(self.args.output_dir, 'worst_predicted.png'))

        # get the best and worst images
        best_best = centre_image_paths[absolute_diffs.argmin()]
        worst_worst = centre_image_paths[absolute_diffs.argmax()]

        print('Best best', best_best)
        print('Worst worst', worst_worst)

        images = {
            best_best: best_3_predicted[best_best],
            worst_worst: worst_3_predicted[worst_worst]
        }

        images[best_best]['filename'] = 'best_best_activations'
        images[worst_worst]['filename'] = 'worst_worst_activations'

        # display the best and worst images' saliency and heatmaps
        self.display_saliency_and_heatmap(images, save=True, display_angles=True)

        # display the saliency of the images at different layers of the neural network
        self.display_saliency(images)

        # display the grad-CAM of the images at different layers of the neural network
        self.display_grad_cam(images)

        # get the labels and x-markers for the data variations
        try:
            variations = ['STRAIGHTS-GF', 'CORNERS-GF', 'STRAIGHTS-1F', 'CORNERS-1F', 'STRAIGHTS-3F']
            x_markers = {variations[i]: validation_combinations[i] for i in range(len(variations))}
        except IndexError:
            x_markers = {}

        # x-points for the graphs
        x_points = [i+1 for i in range(len(predictions))]

        # plot the predictions vs groundtruth with the x-markers
        plot_line_graph([x_points, x_points], [validation_y, predictions],
                        title='Predictions vs Groundtruth',
                        x_label='Number of Images',
                        y_label='Angles',
                        legend=['Groundtruth', 'Predictions'],
                        colours=['green', 'red'],
                        x_markers=x_markers,
                        save_path=os.path.join(self.args.output_dir, 'predictions_vs_groundtruth.png'))

        # plot the absolute differences
        plot_line_graph([x_points], [np.abs(predictions - groundtruth)],
                        title='Absolute Differences between predictions and groundtruth',
                        x_label='Number of Images',
                        y_label='Absolute Differences',
                        colours=['green', 'red'],
                        x_markers=x_markers,
                        save_path=os.path.join(self.args.output_dir, 'absolute_differences.png'))

    def show_images(self, images, save_path=None):
        """Show the images with their predicted and groundtruth angles

        Args:
            images (Dictionary): contains the image names with their predicted and groundtruth values
            save_path (String): path to save the figured images on disk

        Returns:
            None
        """
        # create 1 figure with multiple subplots
        figure = plt.figure()
        rows, columns = 1, 3

        # enumerate through all images
        for i, (k, v) in enumerate(images.items()):
            # extract the prediction and groundtruth angles
            prediction = v['prediction']
            groundtruth = v['groundtruth']

            # read and resize the image
            image = cv2.imread(k, 1)
            image = cv2.resize(image, (400, 400))

            # draw the predicted and groundtruth angles on the image
            image = draw_on_frame(image, prediction, groundtruth, put_text=False)

            # add the image to a subplot
            # add the prediction and groundtruth angles as text
            figure.add_subplot(rows, columns, i+1)
            plt.axis('off')
            plt.title('Prediction: {0:.3f}\nActual: {1:.3f}'.format(math.degrees(prediction),
                                                                    math.degrees(groundtruth)))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # save the figure with the subplots to disk
        if save_path:
            plt.savefig(save_path)

        plt.show()

    def display_saliency(self, images):
        """Displays the saliency of the images at different layers of the CNN

        Visualise activation over layer outputs.
        Saliency - compute the gradient of output category with respect to input image.

        Args:
            images (Dictionary): dictionary of image names with the prediction and groundtruth angles

        Returns:
            None
        """
        """ To visualize activation over layer outputs

        These should contain more detail since they use Conv or Pooling features that contain more spatial detail which
        is lost in Dense layers. The only additional detail compared to saliency is the penultimate_layer_idx.
        This specifies the pre-layer whose gradients should be used

        layer_idx: The layer index within `model.layers` whose filters needs to be visualized.
        filter_indices: filter indices within the layer to be maximized.
            If None, all filters are visualized. (Default value = None)

        :param images:
        :return:
        """

        # picking the layers
        # if the model has too many layers, only use some
        num_layers = (len(self.model.layers))
        if num_layers > 15:
            layers = [i for i in range(num_layers) if i % 50 == 0]
        else:
            layers = [i for i in range(num_layers)]
        num_layers = len(layers)

        # figure with sub-plots
        rows = 5
        columns = (num_layers // rows) + 1

        # loop through the images
        for image_name in images.keys():

            # read and pre-process the image
            image = cv2.imread(image_name, 1)
            preprocessed_image = self.preprocess(image, self.args.crop_to)

            # create a new figure for every image and adjust subplot space
            figure = plt.figure()
            figure.subplots_adjust(wspace=0.25, hspace=0.5)

            # enumerate through all the layers
            for i, layer_index in enumerate(layers):
                # visualise the saliency of the pre-processed image through the layer specified by the index
                # layer_idx = the layer index whose filters need to be visualised
                # filter_indices = filter indices to be maximised (None in this case because 1 output)
                heatmap = visualize_saliency(self.model, layer_idx=layer_index, filter_indices=None,
                                             seed_input=preprocessed_image)

                # convert the colours to a heatmap
                heatmap = np.uint8(cm.jet(heatmap)[..., :3] * 255)

                # create the subplot
                figure.add_subplot(rows, columns, i+1)
                plt.title('Layer {}'.format(i+1))
                plt.axis('off')

                # Overlay is used to alpha blend heatmap onto the image.
                plt.imshow(overlay(cv2.cvtColor(preprocessed_image, cv2.COLOR_YUV2RGB), heatmap, alpha=0.7))

            # save the figure to disk and show
            plt.suptitle('Visualise saliency of all layers')
            plt.savefig(os.path.join(self.args.output_dir, 'saliency-layers-{}.png'.format(image_name.split('/')[-1])))
            plt.show()

    def display_grad_cam(self, images):
        """Display the gradient class activation maps for the input images

        These should contain more detail since they use Conv or Pooling features that contain more spatial detail which
        is lost in Dense layers. The only additional detail compared to saliency is the penultimate_layer_idx.
        This specifies the pre-layer whose gradients should be used

        Args:
            images (Dictionary): dictionary of image names with the prediction and groundtruth angles

        Returns:
            None
        """
        # get the convolution layers within the CNN
        conv_layer_names = [layer.name for layer in self.model.layers if 'conv' in layer.name]

        # reduce the number of conv layers to be tested if too many
        if len(conv_layer_names) > 15:
            conv_layer_names = conv_layer_names[:10]

        columns = 3
        rows = (len(conv_layer_names) // columns) + 1

        for i, image_name in enumerate(images.keys()):
            # read and pre-process each image
            image = cv2.imread(image_name, 1)
            preprocessed_image = self.preprocess(image, self.args.crop_to)

            # plot a figure for each image which will contain sub-plots
            figure = plt.figure()
            figure.subplots_adjust(wspace=0.25, hspace=0.5)

            # loop over all the selected convolutional layers
            for c, conv_layer in enumerate(conv_layer_names):
                # the penultimate layer index is the pre-layer to layer_idx whose feature maps should be used to
                # compute gradients
                penultimate_layer_idx = utils.find_layer_idx(self.model, conv_layer)

                # get the grad-CAM heatmap, layer_idx is always the last layer
                # layer_idx = layer index within `model.layers` whose filters needs to be visualized
                heatmap = visualize_cam(self.model, layer_idx=-1, filter_indices=None, seed_input=preprocessed_image,
                                        penultimate_layer_idx=penultimate_layer_idx)
                # convert the colours to a heatmap
                heatmap = np.uint8(cm.jet(heatmap)[..., :3] * 255)

                # add the subplot and plot the title and axis
                figure.add_subplot(rows, columns, c + 1)
                plt.title('Layer {}'.format(conv_layer))
                plt.axis('off')
                # Overlay is used to alpha blend heatmap onto img.
                plt.imshow(overlay(cv2.cvtColor(preprocessed_image, cv2.COLOR_YUV2RGB), heatmap, alpha=0.7))

            # save the figure
            plt.suptitle('Visualise Grad-CAM')
            plt.savefig(os.path.join(self.args.output_dir, 'grad-cam-layers-{}.png'.format(image_name.split('/')[-1])))
            plt.show()

    def display_saliency_and_heatmap(self, images, save=False, display_angles=False):
        """Display saliency and grad-CAM for the images using gradient modifiers

        Based off:
        https://github.com/raghakot/keras-vis/tree/master/applications/self_driving

        Args:
            images (Dictionary): contains the image paths with their prediction and groundtruth
            save (Boolean): whether to save the results to disk
            display_angles (Boolean): whether to display the angles on the images or not

        Returns:
            None
        """
        # enumerate over all images
        for j, image_name in enumerate(images.keys()):
            # load the image
            image = utils.load_img(image_name)

            # titles for the plots
            titles = ['right steering', 'left steering', 'maintain steering']

            # gradient modifiers
            # negate = activations for negative steering angles (going right)
            # None = positive steering angles (going left)
            # small_values = reduced steering angles (maintaining steering)
            modifiers = ['negate', None, 'small_values']
            functions = [visualize_saliency, visualize_cam]

            columns = 4
            rows = 2

            # preprocess the image
            preprocessed_image = self.preprocess(image, self.args.crop_to)

            # figure for each image, subplots contain the saliency and grad-CAM images
            figure = plt.figure()
            figure.subplots_adjust(wspace=0.25, hspace=0.5)

            subplot_index = 1

            # loop over the saliency and grad-CAM functions
            for f, function in enumerate(functions):

                # plot the original image
                figure.add_subplot(rows, columns, subplot_index)
                plt.imshow(image)

                # show the angles on this original image if necessary
                if display_angles:
                    plt.title('{0}\nPred: {1:.3f}\nActual: {2:.3f}'.format(
                        str(function.__name__),
                        math.degrees(images[image_name]['prediction']),
                        math.degrees(images[image_name]['groundtruth'])
                    ))
                else:
                    plt.title('{}'.format(str(function.__name__)))
                plt.axis('off')

                # loop over each of the gradient modifiers
                subplot_index += 1
                for i, modifier in enumerate(modifiers):
                    # run the function
                    if f == 0:
                        # visualise saliency on the last layer
                        heatmap = function(self.model, layer_idx=-1, filter_indices=None,
                                           seed_input=preprocessed_image, grad_modifier=modifier)
                    else:
                        # visualise the grad-CAM on the second convolutional layer
                        penultimate_layer_idx = utils.find_layer_idx(self.model, 'conv2d_2')
                        heatmap = function(self.model, layer_idx=-1, filter_indices=None,
                                           seed_input=preprocessed_image, grad_modifier=modifier,
                                           penultimate_layer_idx=penultimate_layer_idx)

                    # convert the colours to a heatmap
                    heatmap = np.uint8(cm.jet(heatmap)[..., :3] * 255)

                    # add the subplot for the image
                    figure.add_subplot(rows, columns, subplot_index)
                    plt.title(titles[i])
                    plt.axis('off')
                    subplot_index += 1

                    # overlay is used to alpha blend heatmap onto img.
                    plt.imshow(overlay(cv2.cvtColor(preprocessed_image, cv2.COLOR_YUV2RGB), heatmap, alpha=0.7))

            # save the figure if necessary
            if save:
                plt.savefig(os.path.join(self.args.output_dir,
                                         '{}.png'.format(images[image_name]['filename'])))

            # show the figure
            plt.show()

    def debug(self):
        """Run debugging mode

        Display the saliency and grad-CAM of an image given as input.
        Display the distribution of steering angles in the dataset before and after data augmentation

        Returns:
            None
        """
        # load the model weights from disk
        self.load_model()

        # if an image path is given through the CLI, display the saliency and grad-CAM
        if self.args.image_path:
            # make sure to save the results to disk
            self.display_saliency_and_heatmap({self.args.image_path: {'filename': 'saliency_activation_maps'}},
                                              save=True)

        # split the training data (80/20)
        x_train, x_val, y_train, y_val = self.load_training_data()

        # loop through each combination of images and steering angles in the training data
        new_angles = []
        for images, steering_angle in zip(x_train, y_train):
            # apply the data augmentation and append the new steering angle to a list
            image, steering_angle = self.data_augmentation(images, steering_angle)
            new_angles.append(steering_angle)

        # plot distribution histogram of steering angles before and after augmentation
        plt.title('Distribution of steering angles after data augmentation of training data')
        plt.hist(y_train, label='Before Augmentation', alpha=0.5)
        plt.hist(new_angles, label='After Augmentation', alpha=0.5)
        plt.xlabel('Steering Angle (radians)')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.args.output_dir, 'steering_angle_histogram.png'))
        plt.show()

        # plot some images and steering angles from the batch generator process (augmentation and pre-processing)
        # create figure that will contain 9 subplots
        figure = plt.figure(figsize=(15, 15))
        columns, rows = (3, 3)
        for i in range(1, columns * rows + 1):
            # get a random sample index
            random_index = np.random.randint(0, len(x_train))

            # get the images and steering angle from the sample index
            images = x_train[random_index]
            steering_angle = y_train[random_index]

            # apply augmentation and pre-processing
            image, steering_angle = self.data_augmentation(images, steering_angle)
            image = self.preprocess(image, self.args.crop_to)

            # add the new image as a subplot with the angle in degrees and radians as the title
            a = figure.add_subplot(rows, columns, i)
            plt.imshow(image)
            a.set_title('Rad: {}, Deg: {}'.format(round(steering_angle, 2),
                                                  round(math.degrees(steering_angle), 2)))
            a.set_xticks([])
            a.set_yticks([])

        # save the figure with subplots to disk
        plt.savefig(os.path.join(self.args.output_dir, 'random_augmented_training_images.png'))
        plt.show()

    def print_details(self, epochs, batch_size, training_iterations, test_iterations, num_train_samples, num_val_samples):
        """Print the settings used in the training process

        Also saves the details to disk

        Args:
            epochs (Integer): number of epochs
            batch_size (Integer): size of the batches
            training_iterations (Integer): number of training steps/batches in an epoch
            test_iterations (Integer): number of testing steps/batches in an epoch
            num_train_samples (Integer): number of training samples
            num_val_samples (Integer): number of testing/validation samples

        Returns:
            None
        """
        # create the details string
        details = 'Epochs: {}\nBatch Size: {}\nTraining steps: {}\nValidation steps: {}\n' \
                  'Number training samples: {}\nNumber validation samples: {}\n'.format(
            epochs, batch_size, training_iterations, test_iterations, num_train_samples, num_val_samples
        )

        print(details)

        # write the details to disk in the same output directory
        with open(os.path.join(self.args.output_dir, 'details.txt'), 'w') as f:
            f.write(details)

    def batch_generator(self, image_paths, steering_angles, training):
        """Method that yields a batch size number of images and steering angles to the CNN

        It continuously does this until an epoch finishes, then it resets and does it again.

        Args:
            image_paths (List): 2D list of image paths, 3 per row (centre, left, right)
            steering_angles (List): list of associated steering angles
            training (Boolean): whether doing batch generating for training or validation

        Returns:
            (Tuple): constantly yields batch size images and associated steering angles
        """
        # batch size data to be generated
        batch_size = self.args.batch_size

        # batch size matrices for recording the data
        images = np.empty([batch_size, self.height, self.width, self.channels])
        angles = np.empty(batch_size)

        while True:
            # index for the matrices
            i = 0

            # reset the indexes for the next batch
            # indexes should point to batch size training samples
            if training:
                index_start = self.batch_callback.training_sample_index
                index_end = self.batch_callback.training_sample_index + self.args.batch_size
            else:
                index_start = self.batch_callback.validation_sample_index
                index_end = self.batch_callback.validation_sample_index + self.args.batch_size

            # if training and the beginning of an epoch
            if training and self.batch_callback.begin_epoch:
                # shuffle before beginning of every epoch so different variations of
                # batches are selected
                permutation = np.random.permutation(len(image_paths))
                image_paths = np.asarray(image_paths)[permutation]
                steering_angles = np.asarray(steering_angles)[permutation]
                self.batch_callback.begin_epoch = False

            # for each sample index in the selected batch indexes
            for index in range(index_start, index_end):
                try:
                    # get the centre, left and right images at that index
                    left, center, right = image_paths[index]

                    # get the associated steering angle at that index
                    steering_angle = float(steering_angles[index])

                    # if training, apply random data augmentation
                    if training and np.random.rand() < 0.6:
                        # do data augmentation randomly
                        image, steering_angle = self.data_augmentation([left, center, right], steering_angle)
                    else:
                        # validation - just load centre image
                        image = self.load_image(center)

                    # pre-process the image before adding data to the matrices
                    images[i] = self.preprocess(image, self.args.crop_to)
                    angles[i] = steering_angle

                    i += 1
                except IndexError:
                    # print('Failed')
                    break

            # update the training and validation sample indexes to point to next batch
            if training:
                self.batch_callback.training_sample_index += self.args.batch_size
            else:
                self.batch_callback.validation_sample_index += self.args.batch_size

            # yield the images and angles to the CNN
            yield images, angles

    def data_augmentation(self, images, steering_angle):
        """Wrapper method that performs the data augmentation

        Args:
            images (List): list of image paths
            steering_angle (Float): steering angle

        Returns:
            (Tuple): augmented image and steering angle
        """
        image, steering_angle = self.choose_image(images, steering_angle)
        image, steering_angle = self.random_flip(image, steering_angle)
        image, steering_angle = self.random_translation(image, steering_angle)
        image = self.random_shadow(image)
        image = self.random_brightness(image)

        return image, steering_angle

    def choose_image(self, images, steering_angle):
        """Augmentation step: Randomly choose the image from the centre, left and right images

        Args:
            images (List): List of image paths
            steering_angle (Float): associated steering angle

        Returns:
            (Tuple): Chosen RGB image and associated (altered) steering angle
        """
        # the choice should be random
        choice = np.random.choice(3)

        if choice == 0:
            # left image - right turn
            steering_angle += 0.2
        elif choice == 2:
            # right image - left turn
            steering_angle -= 0.2

        return self.load_image(images[choice]), steering_angle

    def random_flip(self, image, steering_angle):
        """Augmentation step: Apply a random flip to the image

        Args:
            image (List): 3D array representing an RGB image
            steering_angle (Float): steering angle that the image represents

        Returns:
            (Tuple): the altered RGB image and steering angle
        """
        # only flip randomly
        if np.random.rand() < 0.5:
            # flip the image on the vertical axis
            image = cv2.flip(image, 1)

            # negate the steering angle
            steering_angle = -steering_angle

        return image, steering_angle

    def random_translation(self, image, steering_angle, range_x=100, range_y=10):
        """Augmentation step: Apply a random shift to the image horizontally/vertically

        Args:
            image (List): 3D array representing an RGB image
            steering_angle (Float): steering angle that the image represents
            range_x (Integer): how much to translate/shift in the x direction
            range_y (Integer): how much to translate/shift in the y direction

        Returns:
            (Tuple): the altered image and steering angle after the shift
        """
        # get random x and y shifts based on their ranges
        trans_x = range_x * (np.random.rand() - 0.5)
        trans_y = range_y * (np.random.rand() - 0.5)

        # apply the shift to the steering angle
        steering_angle += trans_x * 0.002

        # transform points to another set of points
        # warp the image, keeping the original width and height
        trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
        height, width = image.shape[:2]
        image = cv2.warpAffine(image, trans_m, (width, height))

        return image, steering_angle

    def random_shadow(self, image):
        """Augmentation step: Apply random shadowing to the image

        Args:
            image (List): 3D array representing an RGB image

        Returns:
            (List): 3D array representing an RGB image
        """
        # create 2 points
        x1, y1 = self.width * np.random.rand(), 0  # one arbitrary point at the top of the image
        x2, y2 = self.width * np.random.rand(), self.height  # one arbitrary point at the bottom of the image
        xm, ym = np.mgrid[0:self.height, 0:self.width]  # create mesh grid (horizontal and vertical)

        # mask of zeros with same shape as one of image channels
        mask = np.zeros_like(image[:, :, 1])

        # find all the indexes that match a condition and give them a value of 1 in the matrix (white)
        mask[np.where((ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0)] = 1

        # create a boolean matrix where the mask elements is equal to either 0 or 1
        cond = mask == np.random.randint(2)

        # draw samples from a uniform distribution between 0.2 and 0.5
        s_ratio = np.random.uniform(low=0.2, high=0.5)

        # convert the colour image to HLS colour space
        hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

        # apply the ratio to the indexes of the L channel of the HLS colour space that
        # satisfy the condition of them either being 0 or 1
        hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio

        # convert back to RGB
        return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)

    def random_brightness(self, image):
        """Augmentation step: Apply random brightness changes to the image

        Args:
            image (List): 3D array representing an RGB image

        Returns:
            (List): 3D array representing an RGB image
        """
        # convert the image to the HSV colours space
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # random ratio for the brightness changes
        ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)

        # apply the ratio to the V channel of the HSV colours space which represents the brightness value
        hsv[:, :, 2] = hsv[:, :, 2] * ratio

        # convert back to RGB before returning
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    def preprocess(self, image, crop_to):
        """Pre-processes an image before it is sent to the CNN

        Args:
            image (List): 3D array representing an RGB image
            crop_to (Integer): how much to crop the top of the image by

        Returns:
            (List): preprocessed image
        """
        # crop
        image = image[crop_to:, :, :]

        # resize
        image = cv2.resize(image, (self.width, self.height), cv2.INTER_AREA)

        # convert to YUV
        image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)

        return image

    def start_training(self, x_train, y_train, x_val, y_val):
        """Kickstart the training process

        Args:
            x_train (List): 2D list of images paths, 3 image paths per row representing the training input
            y_train (List): training groundtruth steering angles
            x_val (List): 2D list of images paths, 3 image paths per row representing the validation input
            y_val (List): validation groundtruth steering angles

        Returns:
            None
        """
        # create the training and validation batch generators that will yield batches of images and steering angles
        training_generator = self.batch_generator(x_train, y_train, training=True)
        validation_generator = self.batch_generator(x_val, y_val, training=False)

        # number of complete cycles of the dataset
        epochs = self.args.epochs

        # number of training and validation iterations/steps i.e. number of batches
        train_steps = int(len(x_train) / self.args.batch_size)
        val_steps = int(len(x_val) / self.args.batch_size)

        # print the training details
        self.print_details(epochs=epochs, batch_size=self.args.batch_size, training_iterations=train_steps,
                           test_iterations=val_steps, num_train_samples=len(x_train), num_val_samples=len(x_val))

        # start the training process
        # allows data augmentation on CPU and training on GPU in parallel
        history = self.model.fit_generator(
            training_generator,
            steps_per_epoch=train_steps,
            epochs=epochs,
            max_queue_size=1,
            validation_data=validation_generator,
            validation_steps=val_steps,
            callbacks=self.get_generic_callbacks(),
            verbose=1  # gives useful output information during training
        )

        # save the model history
        with open(os.path.join(self.args.output_dir, 'history.json'), 'w') as f:
            try:
                # keras bug
                for index, lr in enumerate(history.history['lr']):
                    history.history['lr'][index] = float(history.history['lr'][index])
            except KeyError:
                history.history['lr'] = self.args.learning_rate
            json.dump(history.history, f)

        # print the JSON history
        print(history.history)

        # plot the graphs
        self.plot_training_results(history)

    def test_model(self, image):
        """Function to pass an image to a trained model and receive the steering angle

        Args:
            image (List): 3D list representing an RGB image

        Returns:
            (Float): the predicted steering angle from the neural network
        """
        # pre-process image
        image = self.preprocess(image, self.args.crop_to)

        # model expects 4D array
        image = np.array([image])

        # make the prediction and return
        return self.model.predict(image)

    @abstractmethod
    def training(self):
        """Abstract method to be implemented by the sub-class

        Sets up the training process for each model

        Raises:
            NotImplementedError: if the method is not overridden by the derived class
        """
        raise NotImplementedError

    @abstractmethod
    def build_model(self):
        """Abstract method to be implemented by the sub-class

        Builds and returns a models structure

        Raises:
            NotImplementedError: if the method is not overridden by the derived class
        """
        raise NotImplementedError
