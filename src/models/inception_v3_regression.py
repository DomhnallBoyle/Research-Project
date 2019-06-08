"""
    Filename: models/inception_v3_regression.py
    Description: For training, testing and evaluating the InceptionV3 transfer learning regression model
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import cv2
import os
import sys
from keras.layers import Dense, Dropout, Flatten
from keras.models import Input
from keras.optimizers import Adam
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input

# for local source imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# local source imports
from base_model import BaseModel, rmse, r_square

# global constants
CHANNELS, HEIGHT, WIDTH = (3, 75, 200)


class InceptionV3RegressionModel(BaseModel):
    """Extends BaseModel implementing methods for building the InceptionV3 regression model and training it

    Attributes:
        None
    """

    def __init__(self):
        """Instantiating an instance of InceptionV3RegressionModel

        Calls the __init__ of the base class BaseModel
        """
        super().__init__('Transfer Learning using InceptionV3', size=(CHANNELS, HEIGHT, WIDTH), type='regression')

    def training(self):
        """Overridden method for training the model

        This method is specific to every model but the principles are the same.
        Load the data, build the model, compile and start training

        Returns:
            None
        """
        # load the training data (80/20 training, validation split)
        x_train, x_val, y_train, y_val = self.load_training_data()

        # build the model structure
        self.model = self.build_model()

        # minimise mean squared error via gradient descent using Adam optimiser
        self.model.compile(loss='mean_squared_error', optimizer=Adam(
            lr=self.args.learning_rate), metrics=['mae', 'mape', rmse, r_square])

        # call the generic start training method of the base class
        self.start_training(x_train, y_train, x_val, y_val)

    def build_model(self):
        """Overridden method for building the transfer learning InceptionV3 model structure

        Returns:
            model (Sequential): the model structure with the conv and dense layers as well as their settings
        """
        # download the InceptionV3 model weights based on ImageNet, don't include the last few layers
        base_model = InceptionV3(weights='imagenet', include_top=False, input_tensor=Input(shape=(HEIGHT, WIDTH,
                                                                                                  CHANNELS)))

        # freeze all the layers - ensuring there are no changes to the weights of these
        for layer in base_model.layers:
            layer.trainable = False

        # construct the dense layers - similar to NVIDIA model
        # apply dropout, flatten and then dense layers with same activations
        x = base_model.output
        x = Dropout(rate=0.5)(x)
        x = Flatten()(x)
        x = Dense(units=100, activation='elu')(x)
        x = Dense(units=50, activation='elu')(x)
        x = Dense(units=10, activation='elu')(x)
        x = Dense(units=1, activation='elu')(x)

        # construct the model from the base model input and dense layers as output
        model = Model(inputs=base_model.input, outputs=x)

        # print the model summary
        model.summary()

        return model

    def preprocess(self, image, crop_to):
        """Overridden from the BaseModel preprocess method

        Preprocesses an image before it is sent to the CNN

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
    """Create instance of InceptionV3RegressionModel object

    Handle object function based on the CLI arguments

    Returns:
        None
    """
    InceptionV3RegressionModel()


if __name__ == '__main__':
    """Entry point for starting the script
    
    Parses the command line arguments and calls the main function
    
    Example Usage:
    python inception_v3_regression.py training '["/path/to/first/driving_log.csv", "/path/to/second/driving_log.csv"]' 
    <output_dir_path> --crop_to=100 --epochs=2000
    
    python inception_v3_regression.py testing <model_path> <driving_csv_log_path> <output_dir_path> 
    --video_path=/path/to/video.mp4
    
    python inception_v3_regression.py evaluate <model_path> 
    '["/path/to/first/driving_log.csv", "/path/to/second/driving_log.csv"]' <output_dir_path> --crop_to=100
    
    python inception_v3_regression.py debug '["/path/to/first/driving_log.csv", "/path/to/second/driving_log.csv"]' 
    <output_dir_path> --image_path=/path/to/image.jpg --model_path=/path/to/model.h5
    """
    main()
