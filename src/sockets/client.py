"""
    Filename: sockets/client.py
    Description: Contains functionality for creating a client to a server using sockets
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import pickle
from abc import ABC, abstractmethod
from socket import *
from struct import unpack
from threading import Thread


class ClientProtocol(ABC):
    """Abstract base class which creates a socket that binds to an IP and port

    Function is then constantly ran on a thread

    Based on: https://realpython.com/python-sockets/

    Attributes:
        socket (Socket): socket object to connect to server IP and port
        thread (Thread): concurrent thread object to handle the received data
    """

    def __init__(self):
        """Abstract class, cannot be instantiated
        """
        self.socket = None
        self.thread = Thread(target=self.handle)

    def listen(self, server_ip, server_port):
        """Bind to an IP address and port and start the thread to handle sent data

        Args:
            server_ip (String): IP address to bind to
            server_port (Integer): Port to bind to

        Returns:
            None
        """
        self.socket = socket(AF_INET, SOCK_STREAM)
        self.socket.bind((server_ip, server_port))
        self.socket.listen(1)
        self.thread.start()

    def handle(self):
        """Function that is ran in a thread on the socket

        Receives the sent data from the server and runs the handle function on this

        Returns:
            None
        """
        # accept server connection
        (connection, addr) = self.socket.accept()

        # continuously loop until finished
        while True:
            try:
                # receive the incoming length of the message
                bs = connection.recv(8)
                (length,) = unpack('>Q', bs)

                # to record the data sent from the server
                data = b''
                while len(data) < length:
                    # continuously read data from the server in batches rather than all at once until there is no more
                    to_read = length - len(data)
                    data += connection.recv(4096 if to_read > 4096 else to_read)

                # send a token back indicating reading success
                assert len(b'\00') == 1
                connection.sendall(b'\00')

                # load the pickled data back to JSON again
                data = pickle.loads(data)

                # pass the data to the implemented concrete class handle function
                self.handle_function(data)
            except Exception as e:
                # if any issues, print the error, close the socket and exit
                print(e)
                self.close()
                exit()

    @abstractmethod
    def handle_function(self, data):
        """Abstract method to be overridden by the concrete sub-class

        Performs actions on the data sent from the server

        Args:
            data (JSON): data dictionary sent from the server

        Raises:
            NotImplementedError: if the method is not implemented by the base class
        """
        raise NotImplementedError

    def close(self):
        """Close the socket

        Stop the thread and shutdown the socket cleanly

        Returns:
            None
        """
        self.thread.join()
        self.socket.shutdown(SHUT_WR)
        self.socket.close()
        self.socket = None
