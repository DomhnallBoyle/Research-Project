"""
    Filename: models/nvidia.py
    Description: Contains functionality for training, testing and evaluating the NVIDIA regression model
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import os
import sys
from keras.layers import Conv2D, Dense, Dropout, Flatten, Lambda
from keras.models import Sequential
from keras.optimizers import Adam

# for local source imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# local source imports
from base_model import BaseModel, rmse, r_square

# global constants
CHANNELS, HEIGHT, WIDTH = (3, 66, 200)


class NvidiaModel(BaseModel):
    """Extends BaseModel implementing methods for building the NVIDIA model and training it

    Based on:
    https://arxiv.org/abs/1604.07316
    https://github.com/llSourcell/How_to_simulate_a_self_driving_car

    Attributes:
        None
    """

    def __init__(self):
        """Instantiating an instance of NvidiaModel

        Calls the __init__ of the base class BaseModel
        """
        super().__init__('NVIDIA CNN Model', size=(CHANNELS, HEIGHT, WIDTH), type='regression')

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
        self.model.compile(loss='mean_squared_error',
                           optimizer=Adam(lr=self.args.learning_rate),
                           metrics=['mae', 'mape', rmse, r_square])

        # call the generic start training method of the base class
        self.start_training(x_train, y_train, x_val, y_val)

    def build_model(self):
        """Overridden method for building the NVIDIA model structure

        Returns:
            model (Sequential): the model structure with the conv and dense layers as well as their settings
        """
        model = Sequential()

        # image normalisation layer - to avoid saturation and make gradients work better
        # CNN performs optimally when working when small, floating point values are processed
        # appearance of the images are unaltered after this step
        model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(HEIGHT, WIDTH, CHANNELS),
                         output_shape=(HEIGHT, WIDTH, CHANNELS)))

        # 5 convolutional layers
        # uses valid padding - output maps are smaller than the input
        model.add(Conv2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
        model.add(Conv2D(filters=36, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
        model.add(Conv2D(filters=48, kernel_size=(5, 5), strides=(2, 2), activation='elu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))
        model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='elu'))

        # dropout & flatten
        # dropout - randomly selected neurons are ignored during training. Their activation is temporally removed on
        # the forward pass and any weight updates are not applied on the backward pass. Other neurons will have to step
        # in and handle the representation required to make predictions for the missing neurons. The network then
        # becomes less sensitive to the specific weights of neurons resulting in better generalisation and less likely
        # to overfit
        model.add(Dropout(rate=0.5))
        model.add(Flatten())

        # dense layers
        model.add(Dense(units=100, activation='elu'))
        model.add(Dense(units=50, activation='elu'))
        model.add(Dense(units=10, activation='elu'))
        model.add(Dense(units=1))

        # print summary of structure
        model.summary()

        return model


def main():
    """Create instance of NvidiaModel object

    Handle object function based on the CLI arguments

    Returns:
        None
    """
    NvidiaModel()


if __name__ == '__main__':
    """Entry point for starting the script
    
    Calls the main method
    
    Example Usage:
    python nvidia.py training '["/path/to/first/driving_log.csv", "/path/to/second/driving_log.csv"]' 
    <output_dir_path> --crop_to=100 --epochs=2000
    
    python nvidia.py testing <model_path> <driving_csv_log_path> <output_dir_path> --video_path=/path/to/video.mp4
    
    python nvidia.py evaluate <model_path> '["/path/to/first/driving_log.csv", "/path/to/second/driving_log.csv"]' 
    <output_dir_path> --crop_to=100
    
    python nvidia.py debug '["/path/to/first/driving_log.csv", "/path/to/second/driving_log.csv"]' <output_dir_path> 
    --image_path=/path/to/image.jpg --model_path=/path/to/model.h5
    """
    main()
