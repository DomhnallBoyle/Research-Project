"""
    Filename: utils/file_video_stream.py
    Description: Contains functionality for streaming videos in a separate thread
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import cv2
from queue import Queue
from threading import Thread


class FileVideoStream(object):
    """Class that contains functionality for streaming a video from a selected path

    Attributes:
        stream (VideoCapture): video capture object referencing the video stream
        paused (Boolean): indicating if the stream is paused
        stopped (Boolean): indicating if the stream is stopped
        q (Queue): FIFO queue that contains the video frames from the stream
    """

    def __init__(self, path, queue_size=128):
        """Instantiating an instance of FileVideoStream

        Args:
            path (String): path of the video stream
            queue_size (Integer): size of the number of frames that can be in the queue at any one time
        """
        self.stream = cv2.VideoCapture(path)
        self.paused = False
        self.stopped = False
        self.q = Queue(maxsize=queue_size)

    def start(self):
        """Spawns a thread that calls the update method

        Returns:
            (FileVideoStream): the instance of the class itself
        """
        # create an instance of the thread that runs the update method
        # start the thread
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()

        return self

    def update(self):
        """Continually adds frames to the Queue from the stream if there is space to add them

        Returns:
            None
        """
        while True:
            # if the stop method has been called break
            if self.stopped:
                break

            # if the Queue is not full
            if not self.q.full():
                # read a frame from the stream
                success, frame = self.stream.read()

                # if successful and not paused, add the frame to the Queue
                if success and not self.paused:
                    self.q.put(frame)

    def read(self):
        """Reads a frame from the Queue in a FIFO order

        Returns:
            (List): 3D list representing the return RGB frame from the stream
        """
        return self.q.get()

    def more(self):
        """Return True if there are frames in the queue, else False

        Returns:
            (Boolean): if there are frames in the queue or not
        """
        return self.q.qsize() > 0

    def clear(self):
        """Clears the stream by emptying all frames from the Queue

        Raises:
            Exception: if the queue is empty when trying to retrieve a frame. Continue if so

        Returns:
            None
        """
        # loop until the queue is empty
        while not self.q.empty():
            try:
                # get a frame from the queue
                self.q.get(False)
            except Exception:
                # continue if the queue is empty
                continue
            #
            self.q.task_done()

    def stop(self):
        """Stops the stream, sets stopped to True

        Returns:
            None
        """
        self.stopped = True

    def pause(self):
        """Pauses the stream, sets paused to True

        Returns:
            None
        """
        self.paused = True

    def unpause(self):
        """Unpauses the stream, sets paused to False

        Returns:
            None
        """
        self.paused = False
