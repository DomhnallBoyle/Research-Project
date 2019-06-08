"""
    Filename: lane_detection/utils/draw.py
    Description: Contains general drawing functions for displaying images to screens
    Author: Domhnall Boyle
    Maintained by: Domhnall Boyle
    Email: dboyle25@qub.ac.uk
    Python Version: 3.6
"""

# standard and 3rd party library imports
import cv2
import matplotlib.pyplot as plt
import numpy as np

# global constants
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 540


def plt_show_images(images, window_name='Figure', rows=2, columns=2, titles=None):
    """Function to draw multiple figures to screen

    Args:
        images (List): list of images
        window_name (String): name of the window
        rows (Integer): number of rows in the subplot
        columns (Integer): number of columns in the subplot
        titles (List): list of string titles for each figure

    Returns:
        None
    """
    # create titles if none given
    n_images = len(images)
    if titles is None:
        titles = ['Image (%d)' % i for i in range(1, n_images + 1)]

    # create the figure
    fig = plt.figure(figsize=(10, 10))

    # enumerate through each pair of (image, title)
    for n, (image, title) in enumerate(zip(images, titles)):
        # add the subplot
        a = fig.add_subplot(rows, columns, n + 1)

        # make sure plot in grayscale if 2D image
        if image.ndim == 2:
            plt.gray()

        # show the image and set the title
        plt.imshow(image)
        a.set_title(title)

    # set the overall title and show the figure with subplots
    plt.suptitle(window_name)
    plt.show()


def cv2_show_stacked_images(window_name, images):
    """CV2 show images: Stacks images in the 1 Figure

    Images must have same dimensions

    Args:
        window_name (String): name of the window
        images (List): list of RGB images

    Returns:
        None
    """
    # number of rows and columns for the figure
    num_rows = len(images)
    num_columns = len(images[0])

    # get the frame dimensions
    frame_width = IMAGE_WIDTH * num_columns
    frame_height = IMAGE_HEIGHT * num_rows

    # contains the images stacked ontop of eachother
    vstack = []

    # for each row of images
    for row in images:
        # stack the images horizontally
        hstack = []
        for column in row:
            hstack.append(column)

        # vertically stack the horizontal images
        vstack.append(np.hstack(tuple(hstack)))

    # create the final vertical stack, resize it and show it on screen
    frame = np.vstack(tuple(vstack))
    frame = cv2.resize(frame, (frame_width, frame_height))
    cv2.imshow(window_name, frame)


def cv2_show_figured_images(images, titles):
    """Create a separate window for each image and display it

    Args:
        images (List): list of RGB images
        titles (List): list of title strings

    Returns:

    """
    # get the frame dimensions
    frame_width = IMAGE_WIDTH
    frame_height = IMAGE_HEIGHT

    # for each pair of (image, title)
    for n, (image, title) in enumerate(zip(images, titles)):
        # resize the frame and display it using openCV
        image = cv2.resize(image, (frame_width, frame_height))
        cv2.imshow(title, np.uint8(255 * image))


def prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, left_lane, right_lane, offset_meter):
    """Used to put all the advanced lane detection components onto an image

    Adds the binary, birdseye view, left and right lanes onto the image
    Offset and radius text are also added to the image

    Args:
        blend_on_road: color image of lane blend onto the road
        img_binary: thresholded binary image
        img_birdeye: bird's eye view of the thresholded binary image
        img_fit: bird's eye view with detected lane-lines highlighted
        eft_lane: detected left lane-line
        right_lane:  detected right lane-line
        offset_meter: offset from the center of the lane

    Returns:
        (List): 3D list representing an RGB image with all the components on it
    """
    # get the width and height dimensions
    h, w = blend_on_road.shape[:2]

    thumb_ratio = 0.2
    thumb_h, thumb_w = int(thumb_ratio * h), int(thumb_ratio * w)

    off_x, off_y = 20, 15

    # add a gray rectangle to highlight the upper area
    mask = blend_on_road.copy()
    mask = cv2.rectangle(mask, pt1=(0, 0), pt2=(w, thumb_h+2*off_y), color=(0, 0, 0), thickness=cv2.FILLED)
    blend_on_road = cv2.addWeighted(src1=mask, alpha=0.2, src2=blend_on_road, beta=0.8, gamma=0)

    # add thumbnail of binary image
    thumb_binary = cv2.resize(img_binary, dsize=(thumb_w, thumb_h))
    thumb_binary = np.dstack([thumb_binary, thumb_binary, thumb_binary]) * 255
    blend_on_road[off_y:thumb_h+off_y, off_x:off_x+thumb_w, :] = thumb_binary

    # add thumbnail of bird's eye view
    thumb_birdeye = cv2.resize(img_birdeye, dsize=(thumb_w, thumb_h))
    thumb_birdeye = np.dstack([thumb_birdeye, thumb_birdeye, thumb_birdeye]) * 255
    blend_on_road[off_y:thumb_h+off_y, 2*off_x+thumb_w:2*(off_x+thumb_w), :] = thumb_birdeye

    # add thumbnail of bird's eye view (lane-line highlighted)
    thumb_img_fit = cv2.resize(img_fit, dsize=(thumb_w, thumb_h))
    blend_on_road[off_y:thumb_h+off_y, 3*off_x+2*thumb_w:3*(off_x+thumb_w), :] = thumb_img_fit

    # add text (curvature and offset info) on the upper right of the blend
    mean_curvature_meter = np.mean([left_lane.curvature_meter, right_lane.curvature_meter])
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(blend_on_road, 'Curvature radius: {:.02f}m'.format(mean_curvature_meter), (860, 60), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(blend_on_road, 'Offset from center: {:.02f}m'.format(offset_meter), (860, 130), font, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    return blend_on_road
