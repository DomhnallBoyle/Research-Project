"""
    Filename: sockets/server.py
    Description: Contains functionality for creating a server to publish data to clients using sockets
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
from abc import ABC, abstractmethod
from socket import *
from threading import Thread
import time

# global variables
global_image = None


class ServerProtocol(ABC):
    """Abstract base class for extending and creating a server as part of the client-server architecture

    Handles the sending of data to the client

    Attributes:
        socket (Socket): socket object that connects to a specific IP and port
        thread (Thread): thread to be ran concurrently on the socket
    """

    def __init__(self):
        """Abstract class, cannot instantiate
        """
        self.socket = None
        self.thread = Thread(target=self.handle_function)

    def connect(self, server_ip, server_port):
        """Connect to a specific client IP and port

        Starts the thread that handles the sending of the data

        Args:
            server_ip (String): IP address of the client to connect to
            server_port (String): Port of the client application to connect to

        Raises:
            ConnectedRefusedError: if the client has not been started beforehand

        Returns:
            None
        """
        # 10 retries to connect to the IP and port
        retries = 10

        while True:
            # keep trying to connect to the client IP and port until connected or no more tries
            try:
                self.socket = socket(AF_INET, SOCK_STREAM)
                self.socket.connect((server_ip, server_port))
                break
            except ConnectionRefusedError as e:
                # client has not been started to connect to
                print('Start Client First...', e)
                time.sleep(1)
                retries -= 1

            # failed to connect, exit the script
            if retries == 0:
                print('Failed to connect after 10 tries')
                exit()

        # begin the sending thread
        self.thread.start()

    def close(self):
        """Close the socket when finished

        Stops the thead and closes the socket connection

        Returns:
            None
        """
        self.thread.join()
        self.socket.shutdown(SHUT_WR)
        self.socket.close()
        self.socket = None

    @abstractmethod
    def handle_function(self):
        """Abstract function to be ran concurrently in a thread

        Handles the sending of data to the client

        Raises:
            NotImplementedError: if the function is not implemented by the sub-class
        """
        raise NotImplementedError
