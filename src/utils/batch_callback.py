"""
    Filename: utils/batch_callback.py
    Description: Contains functionality for combining images to a video
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import json
import os
from keras.callbacks import Callback


class BatchCallback(Callback):
    """Class that extends Keras Callback class implementing functionality to be completed during the training process

    Saving models during the first epoch, keeping track of validation and training losses

    Attributes:
        save_directory (String): The directory to save data to
        training_sample_index (Integer): Keeps track of the index of the training sample (should go up in batch_size)
        validation_sample_index (Integer): Keeps track of the index of the validation sample
        first_epoch_completed (Boolean): Whether the first epoch has been completed or not
        first_epoch_training_losses (Dictionary): Keeps track of the first epoch training losses
        first_epoch_val_losses (Dictionary): Keeps track of the first epoch validation losses
        begin_epoch (Boolean): Whether or not it is the beginning of an epoch
    """

    def __init__(self, save_directory):
        """Instantiating an instance of BatchCallback

        Calls the __init__ of the superclass Callback

        Args:
            save_directory (String): The directory to save model weights and training/validation losses to
        """
        super().__init__()
        self.save_directory = save_directory
        self.training_sample_index = 0
        self.validation_sample_index = 0
        self.first_epoch_completed = False
        self.first_epoch_training_losses = {}
        self.first_epoch_val_losses = {}
        self.begin_epoch = True

    def on_epoch_end(self, epoch, logs={}):
        """Function to be ran at the end of an epoch

        Overridden from Keras Callback. Saves training/validation losses of the first epoch

        Args:
            epoch (Integer): the current epoch number
            logs (Dictionary): contains logs of the training history during the current epoch

        Returns:
            None
        """
        self.training_sample_index = 0
        self.validation_sample_index = 0
        self.begin_epoch = True

        if not self.first_epoch_completed:
            self.first_epoch_completed = True

            # save training and validation losses
            with open(os.path.join(self.save_directory, 'first_epoch_losses.json'), 'w') as f:
                json.dump({
                    'training_losses': list(self.first_epoch_training_losses.values()),
                    'validation_losses': list(self.first_epoch_val_losses.values())
                }, f)

    def on_train_batch_end(self, batch, logs=None):
        """Function to be ran at the end of a training batch passed through the network

        Args:
            batch (Integer): the current training batch number
            logs (Dictionary): contains logs of the training history during the current epoch

        Returns:
            None
        """
        # if the first epoch hasn't been completed
        if not self.first_epoch_completed:
            # record the training loss after the batch has been passed through the network
            self.first_epoch_training_losses[batch+1] = float(logs.get('loss'))

            # save the model after every 5 batches of the first epoch for evaluation purposes later
            if (batch+1) % 5 == 0:
                model_name = os.path.join(self.save_directory, 'epoch-1-batch-{}.h5'.format(batch+1))
                self.model.save_weights(model_name)

    def on_test_batch_end(self, batch, logs=None):
        """Function to be ran at the end of a test batch passed through the network

        Args:
            batch (Integer): the current batch number
            logs (Dictionary): contains logs of the training history during the current epoch

        Returns:
            None
        """
        # if the first epoch hasn't been completed, record the validation loss for that batch into the dictionary
        # to be saved later
        if not self.first_epoch_completed:
            self.first_epoch_val_losses[batch+1] = float(logs.get('loss'))
