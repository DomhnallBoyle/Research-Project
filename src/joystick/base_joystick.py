"""
    Filename: joystick/base_joystick.py
    Description: Contains functionality for detecting simple lanes from an image
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

from abc import ABC, abstractmethod
import pygame


class BaseJoystick(ABC):
    """
    Base class for PS3 Controller Joystick

    """

    def __init__(self):
        pygame.init()
        self.joystick = None
        self.num_axes = None
        self.num_buttons = None
        self.joystick_inputs = None
        self.joystick_buttons = None

        if self.can_connect():
            self.connect()
        else:
            print('Please connect a joystick...exiting')
            exit(0)

    def can_connect(self):
        joystick_count = pygame.joystick.get_count()
        print('{} joystick/s connected.'.format(joystick_count))

        return joystick_count != 0

    def connect(self):
        self.joystick = pygame.joystick.Joystick(0)
        self.joystick.init()
        self.num_axes = self.joystick.get_numaxes()
        self.num_buttons = self.joystick.get_numbuttons()

    def stop_pressed(self):
        # center button
        if self.joystick_buttons[10] == 1:
            return True

        return False

    def pause_pressed(self):
        # triangle button
        if self.joystick_buttons[2] == 1:
            return True

        return False

    def update_inputs(self):
        self.joystick_inputs = [
            float(self.joystick.get_axis(i)) for i in range(self.num_axes)
        ]

        self.joystick_buttons = [
            self.joystick.get_button(i) for i in range(self.num_buttons)
        ]

    def started_controlling(self):
        self.update_inputs()

        return any(joystick_input != 0 for joystick_input in
                   self.joystick_inputs)

    def get_controls(self, **kwargs):
        self.update_inputs()

        return self._get_controls(**kwargs)

    def normalise(self, value, old_min, old_max, new_min, new_max):
        # https://stats.stackexchange.com/questions/281162/scale-a-number-between-a-range
        return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min

    @abstractmethod
    def _get_controls(self, **kwargs):
        # should return speed and angle
        raise NotImplementedError()
