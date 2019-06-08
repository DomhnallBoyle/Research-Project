#!/usr/bin/env python

# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

# Allows controlling a vehicle with a keyboard. For a simpler and more
# documented example, please take a look at tutorial.py.

"""
    Filename: environment/carla_sim.py
    Description: CARLA client adapted from open-source
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

from __future__ import print_function

# standard and 3rd party library imports
import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('**/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

from carla import ColorConverter as cc

import argparse
import collections
import csv
import cv2
import datetime
import logging
import math
import pickle
import random
import re
import shutil
import weakref
# from cStringIO import StringIO
from struct import pack

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# for local source imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# local source imports
from controller import *
from environment.base_environment import BaseEnvironment
from joystick import CarlaJoystick
from lane_detection import *
from models import *
from sockets import *
from steer_controller import *

# global variables
# left, center, right
global_images = [None, None, None]
global_vehicle = None
global_steer = 0.0
global_pause = True


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate-1] + u'\u2026') if len(name) > truncate else name


class CarlaServerProtocol(ServerProtocol):
    """Class that runs in a thread sending the CARLA data to the client server

    Attributes:
        None
    """

    def __init__(self):
        """Instantiating an instance of CarlaServerProtocol

        Calls the __init__ of the superclass ClientProtocol
        """
        super().__init__()

    def handle_function(self):
        """Overridden method that continuously loops, sending data to the processing node

        Args:
            None

        Returns:
            None
        """
        # global variable for the images from the simulator cameras
        global global_images
        labels = ['Left', 'Center', 'Right']

        while True:
            # if not paused
            if not global_pause:
                # save the global images to a variable
                images = global_images

                # get the vehicle object for the steering angle
                vehicle = global_vehicle

                # make sure images and steering angle is not empty
                if all(image is not None for image in images) and vehicle is not None:
                    # pack the steering angle to a dictionary
                    data = {
                        'steer': vehicle.get_vehicle_control().steer
                    }

                    # enumerate through each RGB image
                    for i, image in enumerate(images):
                        # encode the image and add it to the dictionary
                        _, encoded_image = cv2.imencode('.jpg', image)
                        data[labels[i]] = encoded_image.tostring()

                    # send the length of data off first so the client knows how much
                    # use struct to make sure we have a consistent endianness on the length
                    length = pack('>Q', len(pickle.dumps(data)))
                    # sendall to make sure it blocks if there's back-pressure on the socket
                    self.socket.sendall(length)

                    # send the pickled data across to the client in batches
                    self.socket.sendall(pickle.dumps(data))

                    # success response from the client indicating receipt
                    ack = self.socket.recv(1)


class CarlaClientProtocol(ClientProtocol, BaseEnvironment):
    """Class that runs in a thread obtains the CARLA data from the ServerProtocol

    Extends ClientProtocol and BaseEnvironment

    Attributes:
        args (Object): command line arguments object
        image_counts (Integer): for recording the number of received images
        training_output_directory (String): absolute path of the output directory
        training_output_images (String): absolute path of the images directory
        training_output_csv (String): absolute path of the csv file
    """

    def __init__(self, args):
        """Instantiating an instance of CarlaClientProtocol

        Calls the __init__ of the extended classes
        Runs a specific method based on the run type passed to the CLI

        Args:
            args (Object): command line arguments object
        """
        super().__init__()
        self.args = args
        self.image_counts = [0, 0, 0]
        self.training_output_directory = None
        self.training_output_images = None
        self.training_output_csv = None

        getattr(self, args.run_type)()

    def training(self):
        """Function to be ran before starting the thread during training (collecting data)

        Creates the appropriate directories and files for collecting the data from the simulator

        Returns:
            None
        """
        # setup the absolute paths to the output directories and files
        self.training_output_directory = self.args.output_directory
        self.training_output_images = os.path.join(self.training_output_directory, 'images/')
        self.training_output_csv = os.path.join(self.training_output_directory, 'driving_log.csv')

        # recreate the output directory
        if os.path.exists(self.training_output_directory):
            shutil.rmtree(self.training_output_directory)

        os.makedirs(self.training_output_images)

        # write the header to the CSV file in that directory
        with open(self.training_output_csv, 'w') as f:
            f.write('Center,Left,Right,Angle\n')

    def testing(self):
        """Function to be ran before starting the thread for testing the lane-keeping algorithms

        Gets a specific controller from the command line arguments

        Returns:
            None
        """
        self.get_controller(self.args)

    def handle_function(self, data):
        """Function to handles the data sent from the server protocol

        If in training mode (collecting data), the function will save the data to disk
        If in testing mode, the center image will be decoded and sent to the controller to get a steering angle

        Args:
            data (Dictionary): dictionary of data from the server

        Returns:
            None
        """
        # retrieve the data from the dictionary
        labels = ['Center', 'Left', 'Right']
        images = [data[label] for label in labels]
        steer = data['steer']

        if args.run_type == 'training':
            # TRAINING - save the image and steering angle

            # for the filenames in a row of the CSV file
            image_filenames = []

            for i, image in enumerate(images):
                # get the original images from the bytes array
                image = cv2.imdecode(np.asarray(bytearray(image), dtype=np.uint8), 1)

                # create the relative image filename and save to disk
                image_filename = 'images/{}-{}.jpg'.format(labels[i], self.image_counts[i])
                cv2.imwrite(os.path.join(self.training_output_directory, image_filename), image)

                # append the relative filename to the list
                image_filenames.append(image_filename)

                # update the image counts
                self.image_counts[i] += 1

            # write the row to the CSV containing the relative image paths and steering angle
            with open(self.training_output_csv, 'a') as f:
                f.write('{},{}\n'.format(','.join(image_filenames), steer))

        else:
            # get the original centre image from the bytes array
            image = cv2.imdecode(np.asarray(bytearray(images[1]), dtype=np.uint8), 1)

            # use the controller to get the steering angle
            steering_angle = self.controller.get_steering_angle(image, args.horizon)

            print(steering_angle)

            # set the global variable to the steering angle for the simulator
            global global_steer
            global_steer = steering_angle


class World(object):
    """Class containing functionality for the World which holds references to all actors e.g vehicles, cameras etc

    Taken from open-source
    """

    def __init__(self, carla_world, hud):
        self.world = carla_world
        self.hud = hud
        self.vehicle = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        # self.camera_manager = None
        self.camera_managers = [None, None, None, None]
        self._weather_presets = find_weather_presets()
        self._weather_index = 0
        self.spawn_points = [
            carla.Transform(carla.Location(x=240.4, y=78.1, z=1), carla.Rotation(yaw=92)),
            carla.Transform(carla.Location(x=-143.4, y=-6.4, z=1), carla.Rotation(yaw=-90)),
            carla.Transform(carla.Location(x=106.3, y=58.5, z=1), carla.Rotation(yaw=-161))
        ]
        self.spawn_index = 0
        self.restart()
        self.world.on_tick(hud.on_world_tick)

    def restart(self):
        # Get a random vehicle blueprint.
        blueprint = random.choice(self.world.get_blueprint_library().filter('vehicle.audi.tt'))
        blueprint.set_attribute('role_name', 'hero')
        if blueprint.has_attribute('color'):
            color = random.choice(blueprint.get_attribute('color').recommended_values)
            blueprint.set_attribute('color', color)
        # Spawn the vehicle.
        if self.vehicle is not None:
            spawn_point = self.vehicle.get_transform()
            spawn_point.location.z += 2.0
            spawn_point.rotation.roll = 0.0
            spawn_point.rotation.pitch = 0.0
            self.destroy()
            self.vehicle = self.world.try_spawn_actor(blueprint, spawn_point)
        while self.vehicle is None:
            # SPAWN POINT
            self.vehicle = self.world.try_spawn_actor(blueprint, self.spawn_points[self.spawn_index])

        # Set up the sensors.
        self.collision_sensor = CollisionSensor(self.vehicle, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.vehicle, self.hud)

        # y = how spread out along the bonnet
        # z = height
        # x = distance from front of windscreen
        camera_transforms = [
            carla.Transform(carla.Location(x=1, y=-1.5, z=1.5)),  # left camera
            carla.Transform(carla.Location(x=1, y=0, z=1.5)),  # center camera
            carla.Transform(carla.Location(x=1, y=1.5, z=1.5)),  # right camera
            carla.Transform(carla.Location(x=-6, y=0, z=2.5))  # 3rd person view only
        ]

        for i, camera_manager in enumerate(self.camera_managers):
            cam_index = camera_manager._index if camera_manager is not None else 0
            cam_pos_index = camera_manager._transform_index if camera_manager is not None else 0

            cm = CameraManager(self.vehicle, self.hud, i, camera_transforms[i])
            cm._transform_index = cam_pos_index
            cm.set_sensor(cam_index, notify=False)

            self.camera_managers[i] = cm

        actor_type = get_actor_display_name(self.vehicle)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.vehicle.get_world().set_weather(preset[0])

    def next_spawn_point(self):
        self.spawn_index += 1
        if self.spawn_index == len(self.spawn_points):
            self.spawn_index = 0
        self.vehicle.set_transform(self.spawn_points[self.spawn_index])

    def tick(self, clock):
        self.hud.tick(self, clock)

    def render(self, display):
        # always renders center camera
        self.camera_managers[3].render(display)
        self.hud.render(display)

    def destroy(self):
        actors = [
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.vehicle
        ] + [cm.sensor for cm in self.camera_managers]

        for actor in actors:
            if actor is not None:
                actor.destroy()


class KeyboardControl(object):
    """Class containing functionality for parsing the events of the keyboard and joystick controllers

    Taken from open-source
    """

    def __init__(self, world, start_in_autopilot):
        self._autopilot_enabled = start_in_autopilot
        self._control = carla.VehicleControl()
        self._steer_cache = 0.0
        world.vehicle.set_autopilot(self._autopilot_enabled)
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        try:
            self.joystick = CarlaJoystick()
        except SystemExit:
            self.joystick = None
            pass

    def joystick_control(self, world):
        """Parses the events from the CARLA joystick controller

        Only if the joystick is connected

        Args:
            world (carla.World): the carla world

        Returns:
            None
        """
        # if the joystick is connected
        if self.joystick:

            # get the steering, brake and throttle
            steer, brake, throttle = self.joystick.get_controls()

            # create the vehicle control and apply it to the vehicle
            control = carla.VehicleControl()
            control.steer = steer
            control.throttle = throttle
            control.brake = brake
            world.vehicle.apply_control(control)

            # pause functionality
            global global_pause
            if self.joystick.pause_pressed():
                global_pause = not global_pause
                print('Paused', global_pause)
                time.sleep(0.5)

            # exit if stop pressed
            if self.joystick.stop_pressed():
                print('Stop pressed...exiting')
                exit(0)

            # update the joystick inputs
            self.joystick.update_inputs()

    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    # change camera transformation angle
                    for camera_manager in world.camera_managers:
                        camera_manager.toggle_camera()
                elif event.key == K_s:
                    world.next_spawn_point()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                # elif event.key == K_BACKQUOTE:
                #     world.camera_manager.next_sensor()
                # elif event.key > K_0 and event.key <= K_9:
                #     world.camera_manager.set_sensor(event.key - 1 - K_0)
                # elif event.key == K_r:
                #     world.camera_manager.toggle_recording()
                elif event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.key == K_m:
                    self._control.manual_gear_shift = not self._control.manual_gear_shift
                    self._control.gear = world.vehicle.get_vehicle_control().gear
                    world.hud.notification('%s Transmission' % ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                elif self._control.manual_gear_shift and event.key == K_COMMA:
                    self._control.gear = max(-1, self._control.gear - 1)
                elif self._control.manual_gear_shift and event.key == K_PERIOD:
                    self._control.gear = self._control.gear + 1
                elif event.key == K_p:
                    self._autopilot_enabled = not self._autopilot_enabled
                    world.vehicle.set_autopilot(self._autopilot_enabled)
                    world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))
        if not self._autopilot_enabled and not self.joystick:
            self._parse_keys(pygame.key.get_pressed(), clock.get_time())
            self._control.reverse = self._control.gear < 0
            world.vehicle.apply_control(self._control)

    def _parse_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


class HUD(object):
    """Class containing functionality for the HUD display on the pygame window

    Taken from open-source
    """

    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame_number = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame_number = timestamp.frame_count
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, clock):
        if not self._show_info:
            return
        t = world.vehicle.get_transform()
        v = world.vehicle.get_velocity()
        c = world.vehicle.get_vehicle_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame_number - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16d FPS' % self.server_fps,
            'Client:  % 16d FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.vehicle, truncate=20),
            'Map:     % 20s' % world.world.map_name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'Height:  % 18.0f m' % t.location.z,
            '',
            ('Throttle:', c.throttle, 0.0, 1.0),
            ('Steer:', c.steer, -1.0, 1.0),
            ('Brake:', c.brake, 0.0, 1.0),
            ('Reverse:', c.reverse),
            ('Hand brake:', c.hand_brake),
            ('Manual:', c.manual_gear_shift),
            'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear),
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)
        ]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.vehicle.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))
        self._notifications.tick(world, clock)

        # set the vehicle global
        global global_vehicle
        global_vehicle = world.vehicle

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item: # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


class FadingText(object):
    """Class containing functionality for fading text on the pygame window

    Taken from open-source
    """

    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


class HelpText(object):
    """Class containing functionality to render help text to the pygame screen

    Taken from open-source
    """

    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


class CollisionSensor(object):
    """Class containing functionality for a sensor used to notify about collisions

    Taken from open-source
    """

    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._history = []
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self._history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self._hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self._history.append((event.frame_number, intensity))
        if len(self._history) > 4000:
            self._history.pop(0)


class LaneInvasionSensor(object):
    """Class containing functionality for a sensor that notifies you if you've crossed a line

    Taken from open-source
    """

    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self._hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_detector')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        text = ['%r' % str(x).split()[-1] for x in set(event.crossed_lane_markings)]
        self._hud.notification('Crossed line %s' % ' and '.join(text))


class CameraManager(object):
    """Class containing functionality for creating and controlling the cameras in the simulator

    Taken from open-source
    """

    def __init__(self, parent_actor, hud, global_image_index, transform):
        self.sensor = None
        self._surface = None
        self._parent = parent_actor
        self._hud = hud
        self._recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))
        ]
        self._transform_index = 1
        self._sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette, 'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self._sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '5000')
            item.append(bp)
        self._index = None
        self.global_image_index = global_image_index
        self.camera_transform = transform

    def toggle_camera(self):
        self._transform_index = (self._transform_index + 1) % len(self._camera_transforms)
        # self.sensor.set_transform(self._camera_transforms[self._transform_index])
        # self.sensor.set_transform(carla.Transform(carla.Location(x=1.6, z=1.7)))
        self.sensor.set_transform(self.camera_transform)

    def set_sensor(self, index, notify=True):
        index = index % len(self._sensors)
        needs_respawn = True if self._index is None \
            else self._sensors[index][0] != self._sensors[self._index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self._surface = None

            self.sensor = self._parent.get_world().spawn_actor(
                self._sensors[index][-1],
                self.camera_transform,
                attach_to=self._parent
            )

            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self._hud.notification(self._sensors[index][2])
        self._index = index

    def next_sensor(self):
        self.set_sensor(self._index + 1)

    def toggle_recording(self):
        self._recording = not self._recording
        self._hud.notification('Recording %s' % ('On' if self._recording else 'Off'))

    def render(self, display):
        if self._surface is not None:
            display.blit(self._surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self._sensors[self._index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0]/3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self._hud.dim) / 100.0
            lidar_data += (0.5 * self._hud.dim[0], 0.5 * self._hud.dim[1])
            lidar_data = np.fabs(lidar_data)
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self._hud.dim[0], self._hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self._surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self._sensors[self._index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype('uint8'))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]

            if self.global_image_index != 3:
                global global_images
                global_images[self.global_image_index] = array

            array = array[:, :, ::-1]
            self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))


class CarlaSim(object):
    """Class that provides functionality for starting the game loop, updating and rendering the world

    Attributes:
        args (Object): command line arguments
    """

    def __init__(self, args):
        """Instantiate an instance of the CarlaSim

        Args:
            args (Object): command line arguments
        """
        self.args = args
        self.setup()

    def game_loop(self):
        """Function that runs the game loop updating and rendering the world

        Returns:
            None
        """
        # initialise the pygame environment
        pygame.init()
        pygame.font.init()

        global world
        world = None

        try:
            # set up the CARLA client
            client = carla.Client(self.args.host, self.args.port)
            client.set_timeout(2.0)

            # setup the display
            display = pygame.display.set_mode((self.args.width, self.args.height), pygame.HWSURFACE | pygame.DOUBLEBUF)

            # setup the world objects
            hud = HUD(self.args.width, self.args.height)
            world = World(client.get_world(), hud)
            controller = KeyboardControl(world, self.args.autopilot)

            # setup the clock
            clock = pygame.time.Clock()

            # continuously loop
            while True:
                clock.tick_busy_loop(60)

                # parse the keyboard events
                if controller.parse_events(world, clock):
                    return

                # parse the joystick events
                controller.joystick_control(world)

                # update and render
                world.tick(clock)
                world.render(display)
                pygame.display.flip()

                # if in testing mode, constantly apply a throttle and a steering angle from the controller
                if self.args.run_type == 'testing':
                    world.vehicle.apply_control(carla.VehicleControl(throttle=self.args.throttle, steer=global_steer))
        finally:
            # if anything goes wrong, destroy all actors in the world
            if world is not None:
                if world.vehicle is not None:
                    world.vehicle.destroy()

                world.destroy()

            # destroy the pygame environment
            pygame.quit()

    def setup(self):
        """Run setup before running the game loop

        Returns:
            None
        """
        # if the run type either training or testing
        if self.args.run_type in ['training', 'testing']:
            # setup client-server sockets for handling CARLA data asynchronously
            client = CarlaClientProtocol(self.args)
            client.listen('127.0.0.1', 9999)

            server = CarlaServerProtocol()
            server.connect('127.0.0.1', 9999)


def main(args):
    """Main method that creates an instance of the CarlaSim for controlling the simulator

    Args:
        args (Object): command line arguments

    Returns:
        None
    """
    print(args)

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    carla_sim = CarlaSim(args)

    try:
        carla_sim.game_loop()
    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')


if __name__ == '__main__':
    """Main entry point of the script 
    
    Calls the main method with the CLI arguments
    
    Example Usage: 
    python carla_sim.py training <output_directory_path>
    
    python carla_sim.py testing --controller=modular --lane_detection=simple --steer_controller=trivial
    """
    argparser = argparse.ArgumentParser(description='CARLA Manual Control Client')
    argparser.add_argument('-d', '--debug', action='store_true', dest='debug', help='print debug information')
    argparser.add_argument('--host', metavar='H', default='127.0.0.1', help='IP of the host server (default: '
                                                                            '127.0.0.1)')
    argparser.add_argument('-p', '--port', metavar='P', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    argparser.add_argument('-a', '--autopilot', action='store_true', help='enable autopilot')
    argparser.add_argument('--res', metavar='WIDTHxHEIGHT', default='1280x720', help='window resolution (default: '
                                                                                     '1280x720)')

    subparsers = argparser.add_subparsers(dest='run_type', help='sub-command help')

    parser_training = subparsers.add_parser('training', help='Training help')
    parser_training.add_argument('output_directory', help='Output directory to save to')

    parser_testing = subparsers.add_parser('testing', help='Testing help')
    parser_testing.add_argument('--controller', default='modular')
    parser_testing.add_argument('--lane_detection', default='simple')
    parser_testing.add_argument('--steer_controller', default='trivial')
    parser_testing.add_argument('--model_type', default='nvidia')
    parser_testing.add_argument('--model_path', default=None)
    parser_testing.add_argument('--horizon', default='[[0, 600], [350, 400], [930, 400], [1280, 600]]', type=coords)
    parser_testing.add_argument('--throttle', type=float, default=0.3)

    args = argparser.parse_args()

    main(args)
