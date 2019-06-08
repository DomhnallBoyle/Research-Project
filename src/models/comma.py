"""
    Filename: models/comma.py
    Description: Contains functionality for training, testing and evaluating the Comma AI regression model
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
from keras.layers import Conv2D, Dense, Dropout, ELU, Flatten, Lambda
from keras.models import Sequential

# local source imports
from base_model import BaseModel

# global constants
CHANNELS, HEIGHT, WIDTH = (3, 160, 320)


class CommaModel(BaseModel):
    """Extends BaseModel implementing methods for building the Comma AI model and training it

    Based on:
    https://github.com/commaai/research/blob/master/train_steering_model.py

    Attributes
        None
    """

    def __init__(self):
        """Instantiating an instance of CommaModel

        Calls the __init__ of the base class BaseModel
        """
        super().__init__('Comma.AI CNN Model', size=(CHANNELS, HEIGHT, WIDTH), type='regression')

    def build_model(self):
        """Overridden method for building the NVIDIA model structure

        Returns:
            model (Sequential): the model structure with the conv and dense layers as well as their settings
        """
        model = Sequential()

        # image normalisation layer - to avoid saturation and make gradients work better
        # CNN performs optimally when working when small, floating point values are processed
        # appearance of the images will be unaltered after this step
        model.add(Lambda(lambda x: x/127.5 - 1.0, input_shape=(HEIGHT, WIDTH, CHANNELS),
                         output_shape=(HEIGHT, WIDTH, CHANNELS)))

        # convolutional layers
        # same padding - output is same size as input which means the filter has to go outside the bounds of the image
        model.add(Conv2D(filters=16, kernel_size=(8, 8), strides=(4, 4), padding='same', activation='elu'))
        model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(2, 2), padding='same', activation='elu'))
        model.add(Conv2D(filters=64, kernel_size=(5, 5), strides=(2, 2), padding='same'))

        # flatten the feature maps and apply dropout
        model.add(Flatten())
        model.add(Dropout(rate=0.2))

        model.add(ELU())
        model.add(Dense(units=512))
        model.add(Dropout(rate=0.5))
        model.add(ELU())
        model.add(Dense(units=1))

        # print a model summary
        model.summary()

        return model

    def training(self, args):
        """Overridden method for training the model

        This method is specific to every model but the principles are the same.
        Load the data, build the model, compile and start training

        Due to time constraints, didn't get time to test this model

        Returns:
            None
        """
        pass


def main():
    """Create instance of CommaModel object

    Handle object function based on the CLI arguments

    Returns:
        None
    """
    CommaModel()


if __name__ == '__main__':
    """Entry point for starting the script
    
    Parses the command line arguments and calls the main function
    
    Example Usage:
    python comma.py training '["/path/to/first/driving_log.csv", "/path/to/second/driving_log.csv"]' 
    <output_dir_path> --crop_to=100 --epochs=2000
    
    python comma.py testing <model_path> <driving_csv_log_path> <output_dir_path> --video_path=/path/to/video.mp4
    
    python comma.py evaluate <model_path> '["/path/to/first/driving_log.csv", "/path/to/second/driving_log.csv"]' 
    <output_dir_path> --crop_to=100
    
    python comma.py debug '["/path/to/first/driving_log.csv", "/path/to/second/driving_log.csv"]' <output_dir_path> 
    --image_path=/path/to/image.jpg --model_path=/path/to/model.h5
    """
    main()
