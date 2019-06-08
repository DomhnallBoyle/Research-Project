"""
    Filename: models/inception_v3_classification.py
    Description: For training, testing and evaluating the InceptionV3 classification model
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import cv2
import numpy as np
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import math
import seaborn as sn
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from keras.models import Model
from keras.optimizers import Adam
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# for local source imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# local source imports
from base_model import BaseModel, validation_combinations
from utils import draw_on_frame, plot_line_graph

# global constants
CHANNELS, HEIGHT, WIDTH = (3, 75, 200)
CLASSES = [round(i * 0.1, 1) for i in range(-12, 13)]

# global variables
num_classes = len(CLASSES)


class InceptionV3ClassificationModel(BaseModel):
    """Extends BaseModel implementing methods for building an InceptionV3 classification model and training it

    Attributes:
        None
    """

    def __init__(self):
        """Instantiating an instance of InceptionV3ClassificationModel

        Calls the __init__ of the base class BaseModel
        """
        super().__init__('Transfer Learning using InceptionV3', size=(CHANNELS, HEIGHT, WIDTH), type='classification')

    def load_training_data(self):
        """Overridden method for loading the training data

        This method uses one hot encoding to transform the training labels to binned classifications
        e.g. a steering angle of 0.1 radians = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        where there are 25 bins representing angles from -1.2 - 1.2 radians

        Returns:
            (Tuple): containing the separated training and validation images and groundtruth labels
        """
        # load the training data as normal from the super class
        x_train, x_val, y_train, y_val = super().load_training_data()

        # create the one hot encoder object where the categories are set automatically
        # unknown angles are ignored from the dataset
        encoder = OneHotEncoder(categories='auto', handle_unknown='ignore')

        # enumerate over the training labels - converting them to angles from the classes array
        for i, radian_angle in enumerate(y_train):
            # find the closest index that the angle is to the classes array
            closest_index = np.abs(CLASSES - radian_angle).argmin()
            # set their value to the value from the classes array
            y_train[i] = [CLASSES[closest_index]]

        # enumerate over the testing labels - converting them to angles from the classes array
        for i, radian_angle in enumerate(y_val):
            # find the closest index that the angle is to the classes array
            closest_index = np.abs(CLASSES - radian_angle).argmin()
            # set their value to the value from the classes array
            y_val[i] = [CLASSES[closest_index]]

        # fit and transform the encoder to the training data converting them to bins
        y_train = list(encoder.fit_transform(y_train).toarray())
        # use the same fit on the test data to convert them to bins
        y_val = list(encoder.transform(y_val).toarray())

        # get the number of classes from the validation data for the output of the CNN
        global num_classes
        num_classes = len(y_val[0])

        return x_train, x_val, y_train, y_val

    def training(self):
        """Overridden method for training the model

        This method is specific to every model but the principles are the same.
        Load the data, build the model, compile and start training

        Returns:
            None
        """
        # load the training data (80/20 training, validation split)
        # uses the overridden method which performs one hot encoding of the labels
        x_train, x_val, y_train, y_val = self.load_training_data()

        # build the model structure
        self.model = self.build_model()

        # minimuse the categorical cross-entropy via gradient descent using Adam optimiser
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=Adam(lr=self.args.learning_rate),
                           metrics=['accuracy'])

        # call the generic start training method of the base class
        self.start_training(x_train, y_train, x_val, y_val)

    def build_model(self):
        """Overridden method for building the InceptionV3 model structure

        Transfer learning architecture - freeze the earlier layers and add custom layers

        Returns:
            model (Sequential): the model structure with the
        """
        # download the InceptionV3 model weights based on ImageNet, don't include the last few layers
        base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(HEIGHT, WIDTH, CHANNELS)))

        # freeze all the layers - ensuring there are no changes to the weights of these
        for layer in base_model.layers:
            layer.trainable = False

        # construct the dense layers
        # apply dropout, 1024 neuron dense layer and softmax output layer
        x = base_model.output
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dropout(rate=0.5)(x)
        x = Dense(units=1024, activation='elu')(x)
        predictions = Dense(units=num_classes, activation='softmax')(x)

        # construct the model from the base model input and softmax layer as output
        model = Model(inputs=base_model.input, outputs=predictions)

        # print the model summary
        model.summary()

        return model

    def batch_generator(self, image_paths, steering_angles, training):
        """Overridden batch generator from the base class

        Method that yields a batch size number of images and steering angles to the CNN
        It continually does this until an epoch finishes, then it resets and does it again
        Performs custom batch generating needed for the classification model

        Args:
            image_paths (List): 2D list of image paths, 3 per row (centre, left, right)
            steering_angles (List): list of associated steering angles
            training (Boolean): whether doing batch generating for training or validation

        Returns:
            (Tuple): constantly yields batch size images and associated steering angles
        """
        # batch size to be generated
        batch_size = self.args.batch_size

        # batch size matrices for recording the data
        images = np.empty([batch_size, self.height, self.width, self.channels])
        angles = np.empty([batch_size, num_classes])

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
                    steering_angle = steering_angles[index]

                    # if training, apply random data augmentation
                    if training and np.random.rand() < 0.6:
                        # do data augmentation randomly
                        image, steering_angle = self.data_augmentation([left, center, right], steering_angle)
                    else:
                        # valiation - just load the centre image
                        image = self.load_image(center)

                    # pre-process the image before adding data to the matrices
                    images[i] = np.array([self.preprocess(image, self.args.crop_to)])
                    angles[i] = np.array([steering_angle])

                    i += 1
                except IndexError:
                    # print('Failed')
                    break

            # update the training and validation sample indexes to point to the next batch
            if training:
                self.batch_callback.training_sample_index += self.args.batch_size
            else:
                self.batch_callback.validation_sample_index += self.args.batch_size

            # yield the images and angles to the CNN
            yield images, angles

    def data_augmentation(self, images, steering_angle):
        """Method overridden from the base class - don't apply translation

        Wrapper method that performs the data augmentation

        Args:
            images (List): list of image paths
            steering_angle (Float): steering angle

        Returns:
            (Tuple): augmented image and steering angle
        """
        image, steering_angle = self.choose_image(images, steering_angle)
        image, steering_angle = self.random_flip(image, steering_angle)
        image = self.random_shadow(image)
        image = self.random_brightness(image)

        return image, steering_angle

    def choose_image(self, images, steering_angle):
        """Method overridden from the base class to choose an image

        Augmentation step: Randomly choose an image from the centre, left and right images
        Adjust the steering angle to suit. Deals with one hot encoded angles

        Args:
            images (List): list of image paths
            steering_angle (Float): associated steering angle

        Returns:
            (Tuple): Chosen RGB image and associated (altered) steering angle
        """
        # the choice should be random
        choice = np.random.choice(3)

        # find the index in the one hot encoded array where there is a 1
        index = np.where(steering_angle == 1)[0]

        if choice == 0:
            # left image - right turn
            if index > 0:
                # adjust the previous element to go right (negative)
                steering_angle[index] = 0
                steering_angle[index-1] = 1
        elif choice == 2:
            # right image - left turn
            if index < len(steering_angle) - 1:
                # adjust the next element to go left (positive)
                steering_angle[index] = 0
                steering_angle[index+1] = 1

        return self.load_image(images[choice]), steering_angle

    def random_flip(self, image, steering_angle):
        """Method overridden from base class to choose an image

        Augmentation step: Apply a random flip to the image
        Adjust the steering angle to suit. Deals with one hot encoded angles

        Args:
            image (List): 3D array representing an RGB image
            steering_angle (Float): associated steering angle

        Returns:
            (Tuple): the altered RGB image and steering angle
        """
        # only flip randomly
        if np.random.rand() < 0.5:
            # flip the image on the vertical axis
            image = cv2.flip(image, 1)

            # negate the steering angle
            # find the index of the category index
            index = np.where(steering_angle == 1)[0]

            # set the category to 0
            steering_angle[index] = 0

            # set the flipped category to 1
            steering_angle[(len(steering_angle)-1) - index] = 1

        return image, steering_angle

    def evaluate(self):
        """Overridden method from the base class

        Evaluate the classification model
        Finds the best and worst predicted images as well as draw their saliency and activation maps

        Returns:
            None
        """
        # load the one hot encoded training data
        training_x, validation_x, training_y, validation_y = self.load_training_data()

        # load the model
        self.load_model()

        # storage of the predictions, best and worst predict images
        prediction_classifications = []
        actual_classifications = []
        validation_probabilities = []
        centre_image_paths = []

        # enumerate through the validation data
        for i, images in enumerate(validation_x):
            # grab the centre image and read as RGB
            centre_image_path = images[1]
            centre_image_paths.append(centre_image_path)
            centre_image = cv2.imread(centre_image_path, 1)

            # get the prediction probabilities where length = num_classes
            prediction_probabilities = self.test_model(centre_image)[0]

            # get the index of the maximum probability from the output
            max_probability_index = np.argmax(prediction_probabilities)

            # append the maximum probability to a list
            validation_probabilities.append(prediction_probabilities[max_probability_index])

            # append the predicted classification to a list from the classes arrray
            prediction_classifications.append(str(CLASSES[max_probability_index]))

            # get the actual groundtruth one hot encoded array from the validation
            # append the actual classification for this validation sample to an array
            actual_one_hot_encoding = validation_y[i]
            actual_classifications.append(str(CLASSES[np.argmax(actual_one_hot_encoding)]))

        # make sure that the length of the actual classifications
        assert(len(actual_classifications) == len(prediction_classifications))

        # In a multi-class classification setup, micro-average is preferable if you suspect there might be class
        # imbalance (i.e you may have many more examples of one class than of other classes).
        # in this case, there may be more classes with straight than corner bins
        # get the performance metrics for the classification model
        print('Accuracy', accuracy_score(actual_classifications, prediction_classifications))
        print('Recall', recall_score(actual_classifications, prediction_classifications, average='micro'))
        print('Precision', precision_score(actual_classifications, prediction_classifications, average='micro'))
        print('F1 score', f1_score(actual_classifications, prediction_classifications, average='micro'))

        # represent the classes as strings
        str_classes = [str(cls) for cls in CLASSES]

        # run a confusion matrix between the predicted and groundtruth with the labels
        matrix = confusion_matrix(actual_classifications, prediction_classifications, labels=str_classes)

        # create a dataframe to plot the confusion matrix using seaborn
        # save the diagram to disk
        df_cm = pd.DataFrame(matrix, index=str_classes, columns=str_classes)
        plt.figure(figsize=(20, 20))
        sn.heatmap(df_cm, annot=True, cmap='YlGnBu')
        plt.title('Confusion Matrix')
        plt.xlabel('Predictions')
        plt.ylabel('Groundtruth')
        plt.savefig(os.path.join(self.args.output_dir, 'confusion_matrix.png'))
        plt.show()

        # find the lists to numpy arrays
        actual_classifications = np.asarray(actual_classifications)
        prediction_classifications = np.asarray(prediction_classifications)
        validation_probabilities = np.asarray(validation_probabilities)
        centre_image_paths = np.asarray(centre_image_paths)

        # get the correctly predicted probabilities where the actual classes == predicted classes and vice versa
        correct_probs = validation_probabilities[actual_classifications == prediction_classifications]
        incorrect_probs = validation_probabilities[actual_classifications != prediction_classifications]

        # get the correctly predicted image paths where the actual classes == predicted classes and vice versa
        correct_centre_image_paths = centre_image_paths[actual_classifications == prediction_classifications]
        incorrect_centre_image_paths = centre_image_paths[actual_classifications != prediction_classifications]

        # get the correctly predicted classifications where the actual classes == predicted classes and vice versa
        correct_prediction_classifications = prediction_classifications[actual_classifications == prediction_classifications]
        incorrect_prediction_classifications = prediction_classifications[actual_classifications != prediction_classifications]

        # get the correct actual classifications where the actual classes == predicted classes and vice versa
        correct_actual_classifications = actual_classifications[actual_classifications == prediction_classifications]
        incorrect_actual_classifications = actual_classifications[actual_classifications != prediction_classifications]

        # get the most correct confident and most incorrect confident indexes from the probabilities
        most_confident_correct_indexes = np.argsort(-correct_probs)[:3]
        most_confident_incorrect_indexes = np.argsort(-incorrect_probs)[:3]
        most_confident_correct, most_confident_incorrect = {}, {}

        print(most_confident_correct_indexes)
        print(most_confident_incorrect_indexes)

        # append the most confident correct/incorrect image paths to their respective dictionaries
        for i in range(3):
            # predictions, groundtruth and probabiliies of the most confident correct and incorrect are added too
            most_confident_correct[correct_centre_image_paths[most_confident_correct_indexes[i]]] = {
                'prediction': float(correct_prediction_classifications[most_confident_correct_indexes[i]]),
                'groundtruth': float(correct_actual_classifications[most_confident_correct_indexes[i]]),
                'probability': correct_probs[most_confident_correct_indexes[i]]
            }
            most_confident_incorrect[incorrect_centre_image_paths[most_confident_incorrect_indexes[i]]] = {
                'prediction': float(incorrect_prediction_classifications[most_confident_incorrect_indexes[i]]),
                'groundtruth': float(incorrect_actual_classifications[most_confident_incorrect_indexes[i]]),
                'probability': incorrect_probs[most_confident_incorrect_indexes[i]]
            }

        print('Most confident correct', most_confident_correct)
        print('Most confident incorrect', most_confident_incorrect)

        # show the most confident correct and incorrect images and save them to disk
        self.show_images(most_confident_correct, os.path.join(self.args.output_dir, 'most_confident_correct.png'))
        self.show_images(most_confident_incorrect, os.path.join(self.args.output_dir, 'most_confident_incorrect.png'))

        #############################

        # find the least confident correct and incorrect probabilities
        least_confident_correct_indexes = np.argsort(correct_probs)[:3]
        least_confident_incorrect_indexes = np.argsort(incorrect_probs)[:3]
        least_confident_correct, least_confident_incorrect = {}, {}

        # append the least confident correct/incorrect image paths to their respective dictionaries
        for i in range(3):
            # predictions, groundtruth and probabilities of the least confident correct and incorrect are added too
            least_confident_correct[correct_centre_image_paths[least_confident_correct_indexes[i]]] = {
                'prediction': float(correct_prediction_classifications[least_confident_correct_indexes[i]]),
                'groundtruth': float(correct_actual_classifications[least_confident_correct_indexes[i]]),
                'probability': correct_probs[least_confident_correct_indexes[i]]
            }
            least_confident_incorrect[incorrect_centre_image_paths[least_confident_incorrect_indexes[i]]] = {
                'prediction': float(incorrect_prediction_classifications[least_confident_incorrect_indexes[i]]),
                'groundtruth': float(incorrect_actual_classifications[least_confident_incorrect_indexes[i]]),
                'probability': incorrect_probs[least_confident_incorrect_indexes[i]]
            }

        print('Least confident correct', least_confident_correct)
        print('Least confident incorrect', least_confident_incorrect)

        # show the least confident correct and incorrect images and save them to disk
        self.show_images(least_confident_correct, os.path.join(self.args.output_dir, 'least_confident_correct.png'))
        self.show_images(least_confident_incorrect, os.path.join(self.args.output_dir, 'least_confident_incorrect.png'))

        # only care about the most confident correct and incorrect
        most_most_confident_correct = correct_centre_image_paths[most_confident_correct_indexes[0]]
        most_most_confident_incorrect = incorrect_centre_image_paths[most_confident_incorrect_indexes[0]]

        print('Most most confident correct', most_most_confident_correct)
        print('Most most confident incorrect', most_most_confident_incorrect)

        # append the top most confident correct image path to the dictionary
        # append the top most confident incorrect image path to the dictionary
        images = {
            most_most_confident_correct: most_confident_correct[most_most_confident_correct],
            most_most_confident_incorrect: most_confident_incorrect[most_most_confident_incorrect]
        }

        # append filenames for saving to disk
        images[most_most_confident_correct]['filename'] = 'most_most_confident_correct_activations'
        images[most_most_confident_incorrect]['filename'] = 'most_most_confident_incorrect_activations'

        # display the saliency and grad-CAM activations of these top most confident correct and incorrectly predicted
        self.display_saliency_and_heatmap(images, save=True, display_angles=True)

        # display the saliency of different layers of the network architecture
        self.display_saliency(images)

        # display the grad-CAM of different layers of the network architecture
        self.display_grad_cam(images)

        # markers for the validation data displaying the points of the types of data
        variations = ['STRAIGHTS-GF', 'CORNERS-GF', 'STRAIGHTS-1F', 'CORNERS-1F', 'STRAIGHTS-3F']
        x_markers = {variations[i]: validation_combinations[i] for i in range(len(variations))}

        # convert the prediction and groundtruth classifications as floats
        float_predictions = [float(prediction) for prediction in prediction_classifications]
        float_groundtruth = [float(actual) for actual in actual_classifications]

        # plot the predictions vs groundtruth in a graph with the markers
        x_points = [i+1 for i in range(len(prediction_classifications))]
        plot_line_graph([x_points, x_points], [float_groundtruth, float_predictions],
                        title='Predictions vs Groundtruth',
                        x_label='Number of Images',
                        y_label='Angles',
                        legend=['Groundtruth', 'Predictions'],
                        colours=['green', 'red'],
                        x_markers=x_markers,
                        save_path=os.path.join(self.args.output_dir, 'predictions_vs_groundtruth.png'))

    def show_images(self, images, save_path=None):
        """Show the images with their predicted and groundtruth angles

        Also displays the probability of the images as well as the prediction and groundruth angles

        Args:
            images (Dictionary): contains the image names with their predicted, groundtruth and probability values
            save_path (String): path to save the figured images on disk

        Returns:
            None
        """
        # create 1 figure with multiple subplots
        figure = plt.figure()
        rows, columns = 1, 3

        # enumerate through all the images
        for i, (k, v) in enumerate(images.items()):
            # extract the prediction and groundtruth angles
            prediction = v['prediction']
            groundtruth = v['groundtruth']

            # read and resize the image
            image = cv2.imread(k, 1)
            image = cv2.resize(image, (400, 400))

            # draw the predicted and groundtruth angles on the iamge
            image = draw_on_frame(image, prediction, groundtruth, put_text=False)

            # add the image to a subplot
            # add the prediction and groundtruth angles as text
            figure.add_subplot(rows, columns, i+1)
            plt.axis('off')
            plt.title('Prediction: {0:.3f}\nActual: {1:.3f}\nProbability: {2:.3f}'.format(
                math.degrees(prediction), math.degrees(groundtruth), v['probability']))
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        # save the figure with the subplots to disk
        if save_path:
            plt.savefig(save_path)

        plt.show()

    def debug(self):
        """Overridden method from BaseModel

        Runs debug on the selected model to display the saliency and heatmap of an image
        Also displays the distribution of steering angles before and after data augmentation

        Returns:
            None
        """
        # load the classification model
        self.load_model()

        # display the saliency and grad-CAM heatmap
        if self.args.image_path:
            self.display_saliency_and_heatmap({self.args.image_path: {'filename': 'saliency_activation_maps'}},
                                              save=True)

        # load the training data
        x_train, x_val, y_train, y_val = self.load_training_data()

        # append the steering angles to a list before and after data augmentation
        before_aug = []
        after_aug = []
        for images, steering_angle in zip(x_train, y_train):
            before_aug.append(float(CLASSES[np.argmax(steering_angle)]))
            image, steering_angle = self.data_augmentation(images, steering_angle)
            after_aug.append(float(CLASSES[np.argmax(steering_angle)]))

        # plot distribution of steering angles in a histogram
        plt.title('Distribution of steering angles after data augmentation of training data')
        plt.hist(before_aug, label='Before Augmentation', alpha=0.5)
        plt.hist(after_aug, label='After Augmentation', alpha=0.5)
        plt.xlabel('Steering Angle (radians)')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')
        plt.savefig(os.path.join(self.args.output_dir, 'steering_angle_histogram.png'))
        plt.show()

    def preprocess(self, image, crop_to):
        """Overridden from the BaseModel preprocess method

        Preprocesses an image before it is sent to the CNN
        Applies inceptionV3 pre-processing to the image

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

        # run InceptionV3 preprocessing on the input
        image = preprocess_input(image)

        return image


def main():
    """Create an instance of InceptionV3ClassificationModel

    Handle object function

    Returns:
        None
    """
    InceptionV3ClassificationModel()


if __name__ == '__main__':
    """Entry point for starting the script
    
    Calls the main method
    
    Example Usage:
    python inception_v3_classification.py training '["/path/to/first/driving_log.csv", 
    "/path/to/second/driving_log.csv"]' <output_dir_path> --crop_to=100 --epochs=2000
    
    python inception_v3_classification.py testing <model_path> <driving_csv_log_path> <output_dir_path> 
    --video_path=/path/to/video.mp4
    
    python inception_v3_classification.py evaluate <model_path> 
    '["/path/to/first/driving_log.csv", "/path/to/second/driving_log.csv"]' <output_dir_path> --crop_to=100
    
    python inception_v3_classification.py debug '["/path/to/first/driving_log.csv", "/path/to/second/driving_log.csv"]' 
    <output_dir_path> --image_path=/path/to/image.jpg --model_path=/path/to/model.h5
    """
    main()
