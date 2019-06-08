"""
    Filename: utils/capture_screen.py
    Description: Contains functionality for capturing footage from a screen by grabbing a drawn window
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import autopy
import os
import shutil
import time
import wx
from threading import Thread

# global constants
CURRENT_DIRECTORY = os.path.abspath(os.path.dirname(__file__)) + '/'
OUTPUT_DIRECTORY = CURRENT_DIRECTORY + 'video/'
OFF_X, OFF_Y = 10, 55


def grab_window(coord1, coord2):
    """Function ran in a thread constantly grabbing images from the drawn window

    Saves the images to disk

    Args:
        coord1: (wxWidgets.wxPoint) coordinate 1 of the window
        coord2: (wxWidgets.wxPoint) coordinate 2 of the window

    Returns:
        None
    """
    # countdown before grabbing the screen
    for i in reversed(range(1, 6)):
        print(i)
        time.sleep(1)

    # get the actual x and y coordinates from the position objects
    x1, y1 = coord1.x, coord1.y + OFF_Y
    x2, y2 = coord2.x, coord2.y + OFF_Y

    # get the width and height
    width = x2 - x1
    height = y2 - y1

    # remove output directory if it exists in order to create a new one
    if os.path.exists(OUTPUT_DIRECTORY):
        shutil.rmtree(OUTPUT_DIRECTORY)

    os.mkdir(OUTPUT_DIRECTORY)

    # keep track of the images
    image_number = 1

    # constantly save the images to the output directory
    while True:
        # create the image name
        image_name = OUTPUT_DIRECTORY + '{}.png'.format(image_number)

        # get the image from the autopy library the specific coordinates of the screen
        # and save the image to disk
        autopy.bitmap.capture_screen(
            ((x1 + 2, y1 + 2), (width - 4, height - 4))) \
            .save(image_name)

        # increment the image numbers and add a short delay to the capturing process
        image_number += 1
        time.sleep(0.1)


class TransparentWindow(wx.Frame):
    """Class that extends the wxWidget Frame for drawing a transparent window on screen

    Handles mouse and key events

    Attributes:
        amount (Integer): value for setting the transparency of the window
        timer (wxWidgets.wxTimer): fade timer for the window
        panel (wxWidgets.wxPanel): panel that binds the mouse and key events for drawing on the window
        c1 (wxWidgets.wxPoint): for containing the coordinates of the starting point of the window
        c2 (wxWidgets.wxPoint): for containing the coordinates of the finishing point of the drawn window
    """

    def __init__(self):
        """Instantiating an instance of TransparentWindow

        Calls the __init__ of the superclass wx.Frame with the size of the window screen to draw from
        """
        wx.Frame.__init__(self, None, title='Video Screen Capture', pos=(0, 0),
                          size=(wx.DisplaySize()[0]/2, wx.DisplaySize()[1]))
        self.amount = 5
        self.SetTransparent(self.amount)

        # fade timer
        self.timer = wx.Timer(self, wx.ID_ANY)
        self.timer.Start(60)
        self.Bind(wx.EVT_TIMER, self.alpha_cycle)

        self.SetCursor(wx.Cursor(wx.CURSOR_CROSS))

        self.panel = wx.Panel(self, size=self.GetSize())
        self.panel.Bind(wx.EVT_MOTION, self.on_mouse_move)
        self.panel.Bind(wx.EVT_LEFT_DOWN, self.on_mouse_down)
        self.panel.Bind(wx.EVT_LEFT_UP, self.on_mouse_up)
        self.panel.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_CHAR_HOOK, self.on_key_up)
        self.c1, self.c2 = None, None

    def alpha_cycle(self, event):
        """Bound to an event timer that constantly sets the transparency

        Args:
            event (wx.TimerEvent): TimerEvent that triggers the function

        Returns:
            None
        """
        self.SetTransparent(-100)

    def on_mouse_move(self, event):
        """Function ran when the mouse is moving

        Args:
            event (wx.MouseEvent): Mouse event object when the function is triggered

        Returns:
            None
        """
        # if the mouse is being dragged and the left click is held down
        if event.Dragging() and event.LeftIsDown():
            # update the position of the end coordinate and refresh the UI
            self.c2 = event.GetPosition()
            self.Refresh()

    def on_mouse_down(self, event):
        """Function ran when the mouse is clicked

        Args:
            event (wx.MouseEvent): Mouse event object when the function is triggered

        Returns:
            None
        """
        # update the position of the starting coordinate of the drawn window
        self.c1 = event.GetPosition()

    def on_mouse_up(self, event):
        """Function ran when the mouse click is released

        Args:
            event (wx.MouseEvent): Mouse event object when the function is triggered

        Returns:
            None
        """
        self.SetCursor(wx.Cursor(wx.CURSOR_ARROW))

    def on_paint(self, event):
        """Function bound to the panel to paint the window as it is being drawn

        Args:
            event (wx.PaintEvent): The paint event that triggers the function

        Returns:
            None
        """
        # make sure you have both coordinates set for the window
        if self.c1 is None or self.c2 is None:
            return

        # object to paint the drawn window
        dc = wx.PaintDC(self.panel)

        # draw the rectangle on the window with the coordinates to view the captured window
        dc.DrawRectangle(self.c1.x, self.c1.y, self.c2.x - self.c1.x, self.c2.y - self.c1.y)

    def on_key_up(self, event):
        """Function ran when a key is pressed

        Return pressed: Starts grabbing the images from the window
        Escape pressed: Closes the window

        Args:
            event (wx.KeyEvent): The key event object that triggers the function call

        Returns:
            None
        """
        # get the code of the key pressed
        key_code = event.GetKeyCode()

        if key_code == wx.WXK_RETURN:
            # spawn the capturing thread if enter/return pressed
            self.Hide()
            thread = Thread(target=grab_window, args=(self.c1, self.c2),)
            thread.start()
        elif key_code == wx.WXK_ESCAPE:
            # close the window if escape pressed
            self.Close()


if __name__ == '__main__':
    """Main starting point for the Python script
    
    Creates the transparent window object and starts starts the GUI loop
    
    Example Usage: 
    python capture_screen.py
    """
    app = wx.App(False)
    frm = TransparentWindow()
    frm.Show()
    app.MainLoop()
