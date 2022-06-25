import numpy as np
import cv2
import pandas as pd
import operator
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from keras.utils import Sequence
from config import yolo_config
from PIL import Image
from skimage.transform import resize
import logging
import keyboard
import time
from keyboard_bindings import keyboard_numbers_binding as keyboard_binding
from tkinter import Tk, Label, Button
from tkinter import filedialog
import copy
from scipy.stats import iqr


def initialize_model_logger():
    """"
    The method below sets and initiates the Depth Model logger.
    The model logger tasks are:
    1. To print on the screen different message types (DEBUG, INFO, ERROR, CRITICAL).
    2. Write the messages in the depth_model.log which can be found in logger folder.
    (Created Function)
    """

    # Logging to specific logger which can be configure separately for each script
    logger_model = logging.getLogger(__name__)

    # Set the lowest level of "errors" that you want to see in log.
    logger_model.setLevel(logging.DEBUG)

    # Set the file where the logger messages should be written(relative path).
    dir = os.path.dirname(__file__)
    logger_path = os.path.join(dir, 'logger', 'depth_logger.log')

    # Set the file where the file_handler should write the messages. The "w" means that it deletes everything that
    # was written before (such that the file doesn't get too big)
    file_handler = logging.FileHandler(logger_path, "w")

    # Setting up the message format of the logger as follows:
    # 1. "%(asctime)s:" - Time and date when the message was written/printed.
    # 2. "%(levelname)s" - Text logging level for the message.
    # 3. "%(funcName)s" - Name of the function containing the logging call.
    # 4. "%(message)s" - The logged message.
    formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(funcName)s:%(message)s')

    # Sets the format for the file_handler
    file_handler.setFormatter(formatter)
    logger_model.addHandler(file_handler)

    # Sets the stream_handler and the format for the stream_handler (to print in the console)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger_model.addHandler(stream_handler)

    return logger_model


def clear_keyboard_inputs():
    """"
    Clear the last user input (after processing it)
    (Created Function)
    """
    keyboard.unhook_all()
    keyboard.clear_all_hotkeys()
    time.sleep(0.3)


def select_pretrained_model_for_inference(logger_model):
    """"
    Interface that allows the user to select the pre-trained model on which to run inference.
    (Created Function)
    """
    print(
        "Select the pre-trained model for inference by pressing the keyboard corresponding number (e.g press 1 for NYU):")
    print("1.NYU Depth v2 (indoor scenes)")
    print("2.KITTI (outdoor scenes)")
    while True:

        if keyboard.is_pressed(keyboard_binding[1]):
            logger_model.info("1 was pressed. Loading the NYU model...")
            model_selected_name = 'nyu.h5'
            clear_keyboard_inputs()
            break

        elif keyboard.is_pressed(keyboard_binding[2]):
            logger_model.info("2 was pressed. Loading the KITTI model...")
            model_selected_name = 'kitti.h5'
            clear_keyboard_inputs()
            break

    return model_selected_name


def load_images_using_tkinter(logger_model):
    """
    Import an image or a batch of images (.png only) using tkinter from a user selected file.
    (Created Function)
    """

    # Create the tkinter window
    window = Tk()

    # Add a Label widget
    label = Label(window, text="Press the button below to open a file from which to load the input Image(s)",
                  font=('Aerial 11'))
    label.pack(pady=30)
    # Add the Button Widget
    Button(window, text="Load Image(s) from folder").pack()
    dir = os.path.dirname(__file__)
    # Initial directory from which to select the images
    initial_dir = os.path.join(dir, 'input_images')
    images = filedialog.askopenfilenames(parent=window, initialdir=initial_dir,
                                         title="Load input image(s) from a specific folder",
                                         filetypes=[("png files", "*.png")])

    images = list(images)
    window.destroy()
    logger_model.info("Loaded Images: " + str(images)[:])
    return images

def load_image_using_tkinter(logger_model):
    """
    Import an image (.png only) using tkinter from a user selected file.
    (Created Function)
    """

    # Create the tkinter window
    window = Tk()

    # Add a Label widget
    label = Label(window, text="Press the button below to open a file from which to load the input Image",
                  font=('Aerial 11'))
    label.pack(pady=30)
    # Add the Button Widget
    Button(window, text="Load Image from folder").pack()
    dir = os.path.dirname(__file__)
    # Initial directory from which to select the images
    initial_dir = os.path.join(dir, 'input_images')
    image = filedialog.askopenfilename(parent=window, initialdir=initial_dir,
                                         title="Load input image from a specific folder",
                                         filetypes=[("png files", "*.png")])

    image = [image]
    window.destroy()
    logger_model.info("Loaded Images: " + str(image)[:])
    return image


def load_images(image_files, logger_model):
    """"
    (Modified function)
    -> Added the option to load images of different sizes (before all input images had to be the same size).
    -> The encoder architecture expects the image dimensions to be divisible by 32. Hence for the images where this
    is not the case, we upscale the width/height to the closest number divisible by 32.
    -> Makes sure to convert non-RGB format images to the RGB standard format (3 channels) - Requirement for the model.
      (Palettised colored images can also be used)
    -> Before you were only able to input images with the same width and height. Now it doesn't matter anymore.
    """
    loaded_images = []
    loaded_images_name = []
    for file in image_files:
        x = np.array(Image.open(file))
        parsed_file_name = file.split("/")
        input_image_name = parsed_file_name[-1]
        loaded_images_name.append(input_image_name)
        x = convert_to_rgb_format(x, file, logger_model, input_image_name)
        x = resize_input_image(x, logger_model, input_image_name)
        loaded_images.append(np.stack(x))

    # return np.stack(loaded_images, axis=0)
    return loaded_images, loaded_images_name


def convert_to_rgb_format(image, image_path, logger_model, input_image_name):
    """
    Makes sure to convert non-RGB format images to the RGB standard format (3 channels) - Requirement for the model.
      (Palettised colored images can also be used)
    (Created function)
    """

    # Had a case where a coloured image was "palettised" (2 channels only) - palettised.png in the input_images.
    # Hence in order to solve this issue, we first convert the image to "RGB".
    if image.ndim != 3 or image.shape[-1] != 3:
        image = np.clip(np.asarray(Image.open(image_path).convert('RGB')) / 255, 0, 1)
        logger_model.info(input_image_name + " is not in RGB format. Converting to RGB format...")
    else:
        image = np.clip(np.asarray(Image.open(image_path), dtype=float) / 255, 0, 1)
    return image


def resize_input_image(image, logger_model, input_image_name):
    """"
    The encoder architecture expects the image dimensions to be divisible by 32. Hence for the images where this
    is not the case, we upscale the width/height to the closest number divisible by 32.
    Upscaling method used: biliniar
    (Created Function)
    """
    height , width, channels = image.shape
    input_shape = (height, width)
    output_width = width
    output_height = height
    is_input_image_upscaled = 0

    if width % 32 != 0:
        output_width = width + (32 - width % 32)
        is_input_image_upscaled = 1

    if height % 32 != 0:
        output_height = height + (32 - height % 32)
        is_input_image_upscaled = 1

    output_shape = (output_height, output_width)

    image = resize(image, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True)

    if is_input_image_upscaled == 1:
        logger_model.info(
            input_image_name + " has been upscaled from " + str(input_shape) + " to " + str(output_shape) +
            ".The encoder architecture expects the image dimensions to be divisible by 32.")

    return image


def predict(model, images, logger_model, loaded_input_images_name, minDepth=10, maxDepth=1000, batch_size=1):
    """
    (Modified function)
    -> Now allows the user to load images of different sizes.
    """
    # Support multiple RGBs, one RGB image, even grayscale
    output_images = []
    i = 0
    for image in images:

        logger_model.info("Currently predicting the depth image of " + loaded_input_images_name[i] + " ...")
        if len(image.shape) < 3: image = np.stack((image, image, image), axis=2)
        if len(image.shape) < 4: image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # Compute predictions
        predictions = model.predict(image, batch_size=batch_size)
        # Put in expected range
        output_images.append(np.clip(DepthNorm(predictions, maxDepth=maxDepth), minDepth, maxDepth) / maxDepth)
        i += 1
    return output_images


def output_depth_images(outputs, logger_model, inputs=None, loaded_input_images_name=None):
    """
    (Created function)
    Output the depth images along with the input images to the output_images folder.
    """
    import matplotlib.pyplot as plt
    from skimage.transform import resize
    for index in range(len(outputs)):
        plasma = plt.get_cmap('plasma')

        dir = os.path.dirname(__file__)
        # Output directory for images
        output_dir = os.path.join(dir, 'output_images')

        name_pic = loaded_input_images_name[index].split(".")
        name_pic = name_pic[0]

        # Output the input image
        im1 = Image.fromarray(np.uint8(inputs[index] * 255))
        im1.save(output_dir + "\\" + loaded_input_images_name[index])
        logger_model.info(
            loaded_input_images_name[index] + " finished processing! Image can be found in the output_images folder.")

        rescaled = outputs[index][0][:, :, 0]
        rescaled = rescaled - np.min(rescaled)
        rescaled = rescaled / np.max(rescaled)

        # Output the Colored Depth Image with Original Output Size of the algorithm (w/2, h/2)
        colored_depth_image = np.uint8(plasma(rescaled)[:, :, :3] * 255)
        im2 = Image.fromarray(colored_depth_image)
        colored_depth_image_name = name_pic + "_depth_colored.png"
        im2.save(output_dir + "\\" + colored_depth_image_name)
        logger_model.info(
            colored_depth_image_name + " finished processing! Image can be found in the output_images folder.")

        # Output the "black/white" Depth Image with Original Output Size of the algorithm (w/2, h/2)
        depth_image = np.uint8(to_multichannel(outputs[index][0] * 255))
        im3 = Image.fromarray(depth_image)
        depth_image_name = name_pic + "_depth.png"
        im3.save(output_dir + "\\" + depth_image_name)
        logger_model.info(depth_image_name + " finished processing! Image can be found in the output_images folder.")

        # Output the Colored Depth Image(width, height are the same as the input image)
        upscaled_colored_depth_image_shape = (2 * rescaled.shape[0], 2 * rescaled.shape[1])
        upscaled_colored_depth_image = np.uint8((resize(colored_depth_image, upscaled_colored_depth_image_shape,
                                                        order=3, preserve_range=True, mode='reflect',
                                                        anti_aliasing=True)))
        im4 = Image.fromarray(upscaled_colored_depth_image)
        upscaled_colored_depth_image_name = name_pic + "_upscaled_depth_colored.png"
        im4.save(output_dir + "\\" + upscaled_colored_depth_image_name)
        logger_model.info(
            upscaled_colored_depth_image_name + " finished processing! Image can be found in the output_images folder.")

        # Output the "black/white" Depth Image(width, height are the same as the input image)
        upscaled_depth_image_shape = (2 * outputs[index][0].shape[0], 2 * outputs[index][0].shape[1])
        upscaled_depth_image = np.uint8(to_multichannel(
            resize(outputs[index][0], upscaled_depth_image_shape, order=3, mode='reflect', anti_aliasing=True)) * 255)
        upscaled_depth_image = upscaled_depth_image[:, :, 0]
        # im5 = Image.fromarray(upscaled_depth_image).convert('L')
        im5 = Image.fromarray(upscaled_depth_image)
        upscaled_depth_image_name = name_pic + "_upscaled_depth.png"
        im5.save(output_dir + "\\" + upscaled_depth_image_name)
        logger_model.info(
            upscaled_depth_image_name + " finished processing! Image can be found in the output_images folder.")


def output_upscaled_input_and_depth_image(outputs, logger_model, inputs=None, loaded_input_images_name=None):
    """
    (Created Function)
    Returns the upscaled depth images normalized/not-normalized and input image upscaled
    Writes in the output folder the input and the depth image.
    """

    import matplotlib.pyplot as plt
    from skimage.transform import resize
    for index in range(len(outputs)):

        dir = os.path.dirname(__file__)
        # Output directory for images
        output_dir = os.path.join(dir, 'output_images')

        name_pic = loaded_input_images_name[index].split(".")
        name_pic = name_pic[0]

        # Output the input image
        input_image = np.uint8(inputs[index] * 255)
        im1 = Image.fromarray(input_image)
        im1.save(output_dir + "\\" + loaded_input_images_name[index])
        logger_model.info(
            loaded_input_images_name[index] + " finished processing! Image can be found in the output_images folder.")

        # Output the "black/white" Depth Image(width, height are the same as the input image)
        upscaled_depth_image_shape = (2 * outputs[index][0].shape[0], 2 * outputs[index][0].shape[1])

        upscaled_depth_image_normalized = to_multichannel(resize(outputs[index][0], upscaled_depth_image_shape, order=3, mode='reflect', anti_aliasing=True))
        upscaled_depth_image_normalized = upscaled_depth_image_normalized[:,:,0]

        upscaled_depth_image = np.uint8(to_multichannel(resize(outputs[index][0], upscaled_depth_image_shape, order=3, mode='reflect', anti_aliasing=True)) * 255)

        upscaled_depth_image = upscaled_depth_image[:, :, 0]
        # im5 = Image.fromarray(upscaled_depth_image).convert('L')
        im5 = Image.fromarray(upscaled_depth_image)
        upscaled_depth_image_name = name_pic + "_upscaled_depth.png"
        im5.save(output_dir + "\\" + upscaled_depth_image_name)
        logger_model.info(
            upscaled_depth_image_name + " finished processing! Image can be found in the output_images folder.")

        return input_image, upscaled_depth_image, upscaled_depth_image_normalized


def calculate_distance_to_objects(depth_img_normalized, detections,depth_model_name, logger_model) :
    """
    (Created Function)
    Calculates the distances from the camera to the objects in meters.
    THe depth ROI is selected using the yolo output bounding box (x1,x2,y1,y2).
    The depth ROI image values are normalized from 0 to 1.
    The distance from camera to the object is calculated using the median of the depth image..
    In order to find depth in meters we must multiply the result from the previous step by the maximum distance of the training dataset (10meters for NYU, 80meters for KITTI).
    """

    distances_to_objects = []
    for index, row in detections.iterrows():
        x1, y1, x2, y2, cls, score, w, h = row.values
        depth_img_roi = copy.deepcopy(depth_img_normalized[y1:y2, x1:x2])
        range_model = 0
        # Range in meters for NYU dataset.
        if depth_model_name == 'nyu.h5':
            range_model = 10
        # Range in meters for the Kitti dataset.
        elif depth_model_name == 'kitti.h5':
            range_model = 80

        distance_to_object = calculate_distance_to_object_using_median(depth_img_roi, range_model, index, cls, logger_model)
        distance_to_object = round(distance_to_object, 2)
        distances_to_objects.append(distance_to_object)

    detections["Distance"] = distances_to_objects


    return detections


def calculate_distance_to_object_using_median(depth_img_roi, range_model, index, cls, logger_model):
    """
    Calculates the distance to object in meters using median.
    """

    # Median Depth Image ROI
    depth_median_normalized = np.median(depth_img_roi)
    # Value converted in meters
    depth_median_in_meters = depth_median_normalized * range_model
    logger_model.info("The distance to Object_" + cls + " _ID_" + str(index) + " using median is: " + str(depth_median_in_meters) + " meters.")

    return depth_median_in_meters


def calculate_distance_to_object_using_L2norm_and_median(depth_img_roi, range_model, index, cls, logger_model, cut_off_gain, img):
    """
    Calculates the distance to object using L2 norm (STD) to remove outliers and then using median.
    """

    # Mean Depth Image ROI
    depth_mean_normalized = np.mean(depth_img_roi)

    # Standard Deviation ROI
    depth_l2_normalized = np.std(depth_img_roi)

    # Values converted in meters
    depth_mean = depth_mean_normalized * range_model
    depth_l2 = depth_l2_normalized * range_model

    # User Information
    #logger_model.info("Mean Object_" + cls + " _ID_" + str(index) + ": " + str(depth_mean) + " meters.")
    #logger_model.info("Standard Deviation(L2) Object_" + cls + " _ID_" + str(index) + ": " + str(depth_l2) + " meters.")

    cut_off_std = depth_l2_normalized * cut_off_gain
    lower_std, upper_std = depth_mean_normalized - cut_off_std, depth_mean_normalized + cut_off_std

    img[depth_img_roi < lower_std] = 0
    img[depth_img_roi > upper_std] = 0

    cv2.imshow('New Image', img)
    cv2.waitKey(0)


def output_yolo_image(yolo_image, logger_model, input_image_name, detections_with_distances):
    """
    (Created Function)
    Output the YOLO image in output_images folder.
    """

    dir = os.path.dirname(__file__)
    # Output directory for images
    output_dir = os.path.join(dir, 'output_images')

    input_image_name = input_image_name.split(".")
    name_pic = input_image_name[0]
    yolo_pic_name = name_pic + "_yolo_with_distance.png"
    full_path = output_dir + "\\" + yolo_pic_name
    im = Image.fromarray(yolo_image)
    im.save(full_path)
    logger_model.info(yolo_pic_name + " finished processing! Image can be found in the output_images folder.")


def calculate_distances_from_object(depth_img_normalized, yolo_image, detections, depth_model_name, logger_model):
    """
    (Created Function)
    Calculates the distance from the object in meters.
    The depth ROI image values are normalized from 0 to 1.
    The distance from camera to the object can be calculated in 3 main ways:
    1. Calculating the median of the depth image (ROI).
    2. Removing outliers using L2 norm and then calculating median.
    3. Removing outliers using IQR and then calculating median.
    In order to find depth in meters we must multiply the result from the previous step by the maximum distance of the training dataset (10meters for NYU, 80meters for KITTI).
    """

    test_images = []
    for index, row in detections.iterrows():
        x1, y1, x2, y2, cls, score, w, h = row.values
        depth_img_roi = depth_img_normalized[y1:y2, x1:x2]
        yolo_image_roi = yolo_image[y1:y2, x1:x2]

        # Mean Absolute Deviation
        # mad_l1 = np.mean(np.absolute(depth_img_roi - np.mean(depth_img_roi)))

        # Mean Depth Image ROI
        depth_mean_normalized = np.mean(depth_img_roi)

        # Median Depth Image ROI
        depth_median_normalized = np.median(depth_img_roi)

        # Standard Deviation ROI
        depth_l2_normalized = np.std(depth_img_roi)

        range_model = 0
        if depth_model_name == 'nyu.h5':
            range_model = 10
        elif depth_model_name == 'kitti.h5':
            range_model = 80

        # Values converted to meters
        depth_mean = depth_mean_normalized * range_model
        depth_median = depth_median_normalized * range_model
        # mad_l1 = mad_l1 * range_model
        depth_l2 = depth_l2_normalized * range_model
        # logger_model.info("Mean Absolute Deviation(L1) Object_" + cls + " ID_" + str(index) + ": " + str(mad_l1) + " meters.")
        logger_model.info("Mean Object_" + cls + " _ID_" + str(index) + ": " + str(depth_mean) + " meters.")
        logger_model.info("Median Object_" + cls + " _ID_" + str(index) + ": " + str(depth_median) + " meters.")
        logger_model.info(
            "Standard Deviation(L2) Object_" + cls + " _ID_" + str(index) + ": " + str(depth_l2) + " meters.")

        # Cut-off
        cut_off_std = depth_l2_normalized * 3
        lower_std, upper_std = depth_mean_normalized - cut_off_std, depth_mean_normalized + cut_off_std

        """ WORKS
        # setting threshold of gray image WORKS
        depth_img_roi = np.uint8(depth_img_roi * 255)
        print(depth_img_roi)
        # 10 20 with NYU
        edges = cv2.Canny(depth_img_roi, 10, 20)
        print(edges)
        ret, th2 = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)

        cv2.imshow("img", th2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        """

        """
        depth_img_roi = np.uint8(depth_img_roi * 255)
        print(depth_img_roi)
        edges = cv2.Canny(depth_img_roi, 10, 20)
        #print(edges)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, np.array([]), 30, 200)
        print(lines)
        if lines:
            for line in lines:
               for x1, y1, x2, y2 in line:
                   cv2.line(depth_img_roi, (x1, y1), (x2, y2), (20, 220, 20), 3)

            cv2.imshow('shapes', depth_img_roi)
            cv2.waitKey(0)
        """

        # WORKS
        # setting threshold of gray image WORKS
        depth_img_roi = np.uint8(depth_img_roi * 255)
        print(depth_img_roi)

        # SHOW CANNY FILTER

        ret, th2 = cv2.threshold(depth_img_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow("img", th2)
        cv2.waitKey(0)

        # gray = cv2.blur(depth_img_roi, (3, 3))
        # gray = cv2.bilateralFilter(gray, 11, 17, 17)  # blur. very CPU intensive.
        print(depth_img_roi)
        # cv2.RETR_TREE, cv2.RETR_EXTERNAL
        contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(len(contours))
        image = cv2.drawContours(depth_img_roi, contours, -1, (0, 255, 0), 3)
        cv2.imshow("img", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 10 20 with NYU
        # edges = cv2.Canny(gray, 10, 20)

        # img = auto_canny(depth_img_roi, 0.3)
        # laplacian = cv2.Laplacian(depth_img_roi, cv2.CV_64FC1)

        # print(img)

        # ret, th2 = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)
        # ret, th2  = cv2.threshold(depth_img_roi,0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # th2 = cv2.adaptiveThreshold(depth_img_roi,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
        # cv2.imshow("img", th2)
        # cv2.waitKey(0)

        # CannyThresh = 0.1 * ret
        # edges = cv2.Canny(depth_img_roi, CannyThresh, ret)
        # cv2.imshow("img", edges)
        # cv2.waitKey(0)
        # contours, hierarchy = cv2.findContours(th2, cv2.RETR_EXTERNAL,  cv2.CHAIN_APPROX_SIMPLE)

        # image = cv2.drawContours(depth_img_roi, contours, -1, (0, 255, 0), 2)

        # cv2.imshow("img", image)
        # cv2.waitKey(0)

        cv2.destroyAllWindows()

        # ret, thresh = cv2.threshold(depth_img_roi, 50, 100, cv2.THRESH_BINARY)
        # visualize the binary image
        # cv2.imshow('Binary image', depth_img_roi)
        # cv2.waitKey(0)
        # cv2.imwrite('image_thres1.jpg', thresh)
        # cv2.destroyAllWindows()

        # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE

        """
        _, threshold = cv2.threshold(depth_img_roi, 5, 10, cv2.THRESH)

        # using a findContours() function
        contours, _ = cv2.findContours(
            threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        i = 0

        # list for storing names of shapes
        for contour in contours:

            # here we are ignoring first counter because
            # findcontour function detects whole image as shape
            if i == 0:
                i = 1
                continue

            # cv2.approxPloyDP() function to approximate the shape
            #approx = cv2.approxPolyDP(
            #    contour, 0.01 * cv2.arcLength(contour, True), True)

            # using drawContours() function
            cv2.drawContours(depth_img_roi, [contour], 0, (0, 0, 255), 5)

        cv2.imshow('shapes',depth_img_roi )
        cv2.waitKey(0)


        """
        # q75, q25 = np.percentile(depth_img_roi, [75, 25])
        # iqr = q75 - q25

        # cut_off_iqr = iqr/8
        # lower_iqr, upper_iqr = q25 - cut_off_iqr, q75 + cut_off_iqr

        # low_med, upper_med = depth_median_normalized - (0.1 * depth_mean_normalized), depth_median_normalized + (0.1 *depth_median_normalized)

        # depth_img_roi_copy = copy.deepcopy(depth_img_roi)
        # print(depth_img_roi_copy.shape)

        # yolo_image_roi[depth_img_roi_copy < low_med] = 0
        # yolo_image_roi[depth_img_roi_copy > upper_med] = 0

        # print(yolo_image_roi)

        # mask = np.where(depth_img_roi_copy > lower, depth_img_roi_copy, 0)
        # mask = np.where(mask < upper, 1, 0)
        # print(mask)
        # depth_img_roi_copy[indices] = 0

        # yolo_image_roi[yolo_image_roi > upper] = 0

        # yolo_image_roi[yolo_image_roi < lower] = 0

        # yolo_image_roi[indices] = 0
        test_images.append(yolo_image_roi)

    return test_images



def auto_canny(image, sigma):
	# compute the median of the single channel pixel intensities
	v = np.median(image)
	# apply automatic Canny edge detection using the computed median
	lower = int(max(0, (1.0 - sigma) * v))
	upper = int(min(255, (1.0 + sigma) * v))
	edged = cv2.Canny(image, lower, upper)
	# return the edged image
	return edged


def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Return a sharpened version of the image, using an unsharp mask."""
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    sharpened = float(amount + 1) * image - float(amount) * blurred
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    if threshold > 0:
        low_contrast_mask = np.absolute(image - blurred) < threshold
        np.copyto(sharpened, image, where=low_contrast_mask)
    return sharpened

def output_objects_from_image(images):
    """
    (Created Function)
    Used for testing purposes to see how to improve the boundaries of the object detection.
    """
    i = 0
    for image in images:

        dir = os.path.dirname(__file__)
        # Output directory for images
        output_dir = os.path.join(dir, 'output_images')

        img_name = "imgx_" + str(i) + ".png"
        full_path = output_dir + "\\" + img_name
        im = Image.fromarray(image)
        im.save(full_path)
        i = i+1





def draw_bbox_with_distance(img, detections, cmap, random_color=True, figsize=(10, 10), show_img=True, show_text=True):
    """
    (Modified Function)
    Include the distance to the objects
    Draw bounding boxes on the img.
    :param img: BGR img.
    :param detections: pandas DataFrame containing detections
    :param random_color: assign random color for each objects
    :param cmap: object colormap
    :param plot_img: if plot img with bboxes
    :return: None
    """
    img = np.array(img)
    scale = max(img.shape[0:2]) / 416
    line_width = int(2 * scale)
    print(detections)
    for index, row in detections.iterrows():
        x1, y1, x2, y2, cls, score, w, h, distance = row.values
        color = list(np.random.random(size=3) * 255) if random_color else cmap[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
        if show_text:
            text = f'ID:{index} {cls} {score:.1f} D:{distance}m'
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = max(0.3 * scale, 0.3)
            thickness = max(int(1 * scale), 1)
            (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
            cv2.rectangle(img, (x1 - line_width//2, y1 - text_height), (x1 + text_width, y1), color, cv2.FILLED)
            cv2.putText(img, text, (x1, y1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    if show_img:
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.pause(5)
        plt.show()
    return img



def DepthNorm(x, maxDepth):
    return maxDepth / x


def scale_up(scale, images):
    scaled = []

    for i in range(len(images)):
        img = images[i]
        output_shape = (scale * img.shape[0], scale * img.shape[1])
        scaled.append(resize(img, output_shape, order=1, preserve_range=True, mode='reflect', anti_aliasing=True))

    return np.stack(scaled)


def to_multichannel(i):
    if i.shape[2] == 3: return i
    i = i[:, :, 0]
    return np.stack((i, i, i), axis=2)


def display_images(outputs, inputs=None, gt=None, is_colormap=True, is_rescale=True):
    import matplotlib.pyplot as plt
    import skimage
    from skimage.transform import resize

    plasma = plt.get_cmap('plasma')

    shape = (outputs[0].shape[0], outputs[0].shape[1], 3)

    all_images = []

    for i in range(outputs.shape[0]):
        imgs = []

        if isinstance(inputs, (list, tuple, np.ndarray)):
            x = to_multichannel(inputs[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True)
            imgs.append(x)

        if isinstance(gt, (list, tuple, np.ndarray)):
            x = to_multichannel(gt[i])
            x = resize(x, shape, preserve_range=True, mode='reflect', anti_aliasing=True)
            imgs.append(x)

        if is_colormap:
            rescaled = outputs[i][:, :, 0]
            if is_rescale:
                rescaled = rescaled - np.min(rescaled)
                rescaled = rescaled / np.max(rescaled)
            imgs.append(plasma(rescaled)[:, :, :3])
        else:
            imgs.append(to_multichannel(outputs[i]))

        img_set = np.hstack(imgs)
        all_images.append(img_set)

    all_images = np.stack(all_images)

    return skimage.util.montage(all_images, multichannel=True, fill=(0, 0, 0))


def save_images(filename, outputs, inputs=None, gt=None, is_colormap=True, is_rescale=False):
    montage = display_images(outputs, inputs, is_colormap, is_rescale)
    im = Image.fromarray(np.uint8(montage * 255))
    im.save(filename)


def load_test_data(test_data_zip_file='nyu_test.zip'):
    print('Loading test data...', end='')
    import numpy as np
    from data import extract_zip
    data = extract_zip(test_data_zip_file)
    from io import BytesIO
    rgb = np.load(BytesIO(data['eigen_test_rgb.npy']))
    depth = np.load(BytesIO(data['eigen_test_depth.npy']))
    crop = np.load(BytesIO(data['eigen_test_crop.npy']))
    print('Test data loaded.\n')
    return {'rgb': rgb, 'depth': depth, 'crop': crop}


def compute_errors(gt, pred):
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    abs_rel = np.mean(np.abs(gt - pred) / gt)
    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())
    log_10 = (np.abs(np.log10(gt) - np.log10(pred))).mean()
    return a1, a2, a3, abs_rel, rmse, log_10


def evaluate(model, rgb, depth, crop, batch_size=6, verbose=False):
    N = len(rgb)

    bs = batch_size

    predictions = []
    testSetDepths = []

    for i in range(N // bs):
        x = rgb[(i) * bs:(i + 1) * bs, :, :, :]

        # Compute results
        true_y = depth[(i) * bs:(i + 1) * bs, :, :]
        pred_y = scale_up(2, predict(model, x / 255, minDepth=10, maxDepth=1000, batch_size=bs)[:, :, :, 0]) * 10.0

        # Test time augmentation: mirror image estimate
        pred_y_flip = scale_up(2,
                               predict(model, x[..., ::-1, :] / 255, minDepth=10, maxDepth=1000, batch_size=bs)[:, :, :,
                               0]) * 10.0

        # Crop based on Eigen et al. crop
        true_y = true_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
        pred_y = pred_y[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]
        pred_y_flip = pred_y_flip[:, crop[0]:crop[1] + 1, crop[2]:crop[3] + 1]

        # Compute errors per image in batch
        for j in range(len(true_y)):
            predictions.append((0.5 * pred_y[j]) + (0.5 * np.fliplr(pred_y_flip[j])))
            testSetDepths.append(true_y[j])

    predictions = np.stack(predictions, axis=0)
    testSetDepths = np.stack(testSetDepths, axis=0)

    e = compute_errors(predictions, testSetDepths)

    if verbose:
        print("{:>10}, {:>10}, {:>10}, {:>10}, {:>10}, {:>10}".format('a1', 'a2', 'a3', 'rel', 'rms', 'log_10'))
        print("{:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(e[0], e[1], e[2], e[3], e[4], e[5]))

    return e


def load_weights(model, weights_file_path):
    conv_layer_size = 110
    conv_output_idxs = [93, 101, 109]
    with open(weights_file_path, 'rb') as file:
        major, minor, revision, seen, _ = np.fromfile(file, dtype=np.int32, count=5)

        bn_idx = 0
        for conv_idx in range(conv_layer_size):
            conv_layer_name = f'conv2d_{conv_idx}' if conv_idx > 0 else 'conv2d'
            bn_layer_name = f'batch_normalization_{bn_idx}' if bn_idx > 0 else 'batch_normalization'

            conv_layer = model.get_layer(conv_layer_name)
            filters = conv_layer.filters
            kernel_size = conv_layer.kernel_size[0]
            input_dims = conv_layer.input_shape[-1]

            if conv_idx not in conv_output_idxs:
                # darknet bn layer weights: [beta, gamma, mean, variance]
                bn_weights = np.fromfile(file, dtype=np.float32, count=4 * filters)
                # tf bn layer weights: [gamma, beta, mean, variance]
                bn_weights = bn_weights.reshape((4, filters))[[1, 0, 2, 3]]
                bn_layer = model.get_layer(bn_layer_name)
                bn_idx += 1
            else:
                conv_bias = np.fromfile(file, dtype=np.float32, count=filters)

            # darknet shape: (out_dim, input_dims, height, width)
            # tf shape: (height, width, input_dims, out_dim)
            conv_shape = (filters, input_dims, kernel_size, kernel_size)
            conv_weights = np.fromfile(file, dtype=np.float32, count=np.product(conv_shape))
            conv_weights = conv_weights.reshape(conv_shape).transpose([2, 3, 1, 0])

            if conv_idx not in conv_output_idxs:
                conv_layer.set_weights([conv_weights])
                bn_layer.set_weights(bn_weights)
            else:
                conv_layer.set_weights([conv_weights, conv_bias])

        if len(file.read()) == 0:
            print('all weights read')
        else:
            print(f'failed to read  all weights, # of unread weights: {len(file.read())}')


def get_detection_data(img, model_outputs, class_names):
    """

    :param img: target raw image
    :param model_outputs: outputs from inference_model
    :param class_names: list of object class names
    :return:
    """

    num_bboxes = model_outputs[-1][0]
    boxes, scores, classes = [output[0][:num_bboxes] for output in model_outputs[:-1]]

    h, w = img.shape[:2]
    df = pd.DataFrame(boxes, columns=['x1', 'y1', 'x2', 'y2'])
    df[['x1', 'x2']] = (df[['x1', 'x2']] * w).astype('int64')
    df[['y1', 'y2']] = (df[['y1', 'y2']] * h).astype('int64')
    df['class_name'] = np.array(class_names)[classes.astype('int64')]
    df['score'] = scores
    df['w'] = df['x2'] - df['x1']
    df['h'] = df['y2'] - df['y1']

    #print(f'# of bboxes: {num_bboxes}')
    return df

def read_annotation_lines(annotation_path, test_size=None, random_seed=5566):
    with open(annotation_path) as f:
        lines = f.readlines()
    if test_size:
        return train_test_split(lines, test_size=test_size, random_state=random_seed)
    else:
        return lines



def draw_bbox(img, detections, cmap, random_color=True, figsize=(10, 10), show_img=True, show_text=True):
    """
    (Modified Function)
    Draw bounding boxes on the img.
    :param img: BGR img.
    :param detections: pandas DataFrame containing detections
    :param random_color: assign random color for each objects
    :param cmap: object colormap
    :param plot_img: if plot img with bboxes
    :return: None
    """
    img = np.array(img)
    scale = max(img.shape[0:2]) / 416
    line_width = int(2 * scale)
    for index, row in detections.iterrows():
        x1, y1, x2, y2, cls, score, w, h = row.values
        color = list(np.random.random(size=3) * 255) if random_color else cmap[cls]
        cv2.rectangle(img, (x1, y1), (x2, y2), color, line_width)
        if show_text:
            text = f'ID:{index} {cls} {score:.2f}'
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = max(0.3 * scale, 0.3)
            thickness = max(int(1 * scale), 1)
            (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=thickness)[0]
            cv2.rectangle(img, (x1 - line_width//2, y1 - text_height), (x1 + text_width, y1), color, cv2.FILLED)
            cv2.putText(img, text, (x1, y1), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    if show_img:
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.pause(5)
        plt.show()
    return img


class DataGenerator(Sequence):
    """
    Generates data for Keras
    ref: https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    """
    def __init__(self,
                 annotation_lines,
                 class_name_path,
                 folder_path,
                 max_boxes=100,
                 shuffle=True):
        self.annotation_lines = annotation_lines
        self.class_name_path = class_name_path
        self.num_classes = len([line.strip() for line in open(class_name_path).readlines()])
        self.num_gpu = yolo_config['num_gpu']
        self.batch_size = yolo_config['batch_size'] * self.num_gpu
        self.target_img_size = yolo_config['img_size']
        self.anchors = np.array(yolo_config['anchors']).reshape((9, 2))
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.annotation_lines))
        self.folder_path = folder_path
        self.max_boxes = max_boxes
        self.on_epoch_end()

    def __len__(self):
        'number of batches per epoch'
        return int(np.ceil(len(self.annotation_lines) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate indexes of the batch
        idxs = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        lines = [self.annotation_lines[i] for i in idxs]

        # Generate data
        X, y_tensor, y_bbox = self.__data_generation(lines)

        return [X, *y_tensor, y_bbox], np.zeros(len(lines))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, annotation_lines):
        """
        Generates data containing batch_size samples
        :param annotation_lines:
        :return:
        """

        X = np.empty((len(annotation_lines), *self.target_img_size), dtype=np.float32)
        y_bbox = np.empty((len(annotation_lines), self.max_boxes, 5), dtype=np.float32)  # x1y1x2y2

        for i, line in enumerate(annotation_lines):
            img_data, box_data = self.get_data(line)
            X[i] = img_data
            y_bbox[i] = box_data

        y_tensor, y_true_boxes_xywh = preprocess_true_boxes(y_bbox, self.target_img_size[:2], self.anchors, self.num_classes)

        return X, y_tensor, y_true_boxes_xywh

    def get_data(self, annotation_line):
        line = annotation_line.split()
        img_path = line[0]
        img = cv2.imread(os.path.join(self.folder_path, img_path))[:, :, ::-1]
        ih, iw = img.shape[:2]
        h, w, c = self.target_img_size
        boxes = np.array([np.array(list(map(float, box.split(',')))) for box in line[1:]], dtype=np.float32) # x1y1x2y2
        scale_w, scale_h = w / iw, h / ih
        img = cv2.resize(img, (w, h))
        image_data = np.array(img) / 255.

        # correct boxes coordinates
        box_data = np.zeros((self.max_boxes, 5))
        if len(boxes) > 0:
            np.random.shuffle(boxes)
            boxes = boxes[:self.max_boxes]
            boxes[:, [0, 2]] = boxes[:, [0, 2]] * scale_w  # + dx
            boxes[:, [1, 3]] = boxes[:, [1, 3]] * scale_h  # + dy
            box_data[:len(boxes)] = boxes

        return image_data, box_data


def preprocess_true_boxes(true_boxes, input_shape, anchors, num_classes):
    '''Preprocess true boxes to training input format

    Parameters
    ----------
    true_boxes: array, shape=(bs, max boxes per img, 5)
        Absolute x_min, y_min, x_max, y_max, class_id relative to input_shape.
    input_shape: array-like, hw, multiples of 32
    anchors: array, shape=(N, 2), (9, wh)
    num_classes: int

    Returns
    -------
    y_true: list of array, shape like yolo_outputs, xywh are reletive value

    '''

    num_stages = 3  # default setting for yolo, tiny yolo will be 2
    anchor_mask = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
    bbox_per_grid = 3
    true_boxes = np.array(true_boxes, dtype='float32')
    true_boxes_abs = np.array(true_boxes, dtype='float32')
    input_shape = np.array(input_shape, dtype='int32')
    true_boxes_xy = (true_boxes_abs[..., 0:2] + true_boxes_abs[..., 2:4]) // 2  # (100, 2)
    true_boxes_wh = true_boxes_abs[..., 2:4] - true_boxes_abs[..., 0:2]  # (100, 2)

    # Normalize x,y,w, h, relative to img size -> (0~1)
    true_boxes[..., 0:2] = true_boxes_xy/input_shape[::-1]  # xy
    true_boxes[..., 2:4] = true_boxes_wh/input_shape[::-1]  # wh

    bs = true_boxes.shape[0]
    grid_sizes = [input_shape//{0:8, 1:16, 2:32}[stage] for stage in range(num_stages)]
    y_true = [np.zeros((bs,
                        grid_sizes[s][0],
                        grid_sizes[s][1],
                        bbox_per_grid,
                        5+num_classes), dtype='float32')
              for s in range(num_stages)]
    # [(?, 52, 52, 3, 5+num_classes) (?, 26, 26, 3, 5+num_classes)  (?, 13, 13, 3, 5+num_classes) ]
    y_true_boxes_xywh = np.concatenate((true_boxes_xy, true_boxes_wh), axis=-1)
    # Expand dim to apply broadcasting.
    anchors = np.expand_dims(anchors, 0)  # (1, 9 , 2)
    anchor_maxes = anchors / 2.  # (1, 9 , 2)
    anchor_mins = -anchor_maxes  # (1, 9 , 2)
    valid_mask = true_boxes_wh[..., 0] > 0  # (1, 100)

    for batch_idx in range(bs):
        # Discard zero rows.
        wh = true_boxes_wh[batch_idx, valid_mask[batch_idx]]  # (# of bbox, 2)
        num_boxes = len(wh)
        if num_boxes == 0: continue
        wh = np.expand_dims(wh, -2)  # (# of bbox, 1, 2)
        box_maxes = wh / 2.  # (# of bbox, 1, 2)
        box_mins = -box_maxes  # (# of bbox, 1, 2)

        # Compute IoU between each anchors and true boxes for responsibility assignment
        intersect_mins = np.maximum(box_mins, anchor_mins)  # (# of bbox, 9, 2)
        intersect_maxes = np.minimum(box_maxes, anchor_maxes)
        intersect_wh = np.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_area = np.prod(intersect_wh, axis=-1)  # (9,)
        box_area = wh[..., 0] * wh[..., 1]  # (# of bbox, 1)
        anchor_area = anchors[..., 0] * anchors[..., 1]  # (1, 9)
        iou = intersect_area / (box_area + anchor_area - intersect_area)  # (# of bbox, 9)

        # Find best anchor for each true box
        best_anchors = np.argmax(iou, axis=-1)  # (# of bbox,)
        for box_idx in range(num_boxes):
            best_anchor = best_anchors[box_idx]
            for stage in range(num_stages):
                if best_anchor in anchor_mask[stage]:
                    x_offset = true_boxes[batch_idx, box_idx, 0]*grid_sizes[stage][1]
                    y_offset = true_boxes[batch_idx, box_idx, 1]*grid_sizes[stage][0]
                    # Grid Index
                    grid_col = np.floor(x_offset).astype('int32')
                    grid_row = np.floor(y_offset).astype('int32')
                    anchor_idx = anchor_mask[stage].index(best_anchor)
                    class_idx = true_boxes[batch_idx, box_idx, 4].astype('int32')
                    # y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 0] = x_offset - grid_col  # x
                    # y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 1] = y_offset - grid_row  # y
                    # y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, :4] = true_boxes_abs[batch_idx, box_idx, :4] # abs xywh
                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, :2] = true_boxes_xy[batch_idx, box_idx, :]  # abs xy
                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 2:4] = true_boxes_wh[batch_idx, box_idx, :]  # abs wh
                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 4] = 1  # confidence

                    y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 5+class_idx] = 1  # one-hot encoding
                    # smooth
                    # onehot = np.zeros(num_classes, dtype=np.float)
                    # onehot[class_idx] = 1.0
                    # uniform_distribution = np.full(num_classes, 1.0 / num_classes)
                    # delta = 0.01
                    # smooth_onehot = onehot * (1 - delta) + delta * uniform_distribution
                    # y_true[stage][batch_idx, grid_row, grid_col, anchor_idx, 5:] = smooth_onehot

    return y_true, y_true_boxes_xywh

"""
 Calculate the AP given the recall and precision array
    1st) We compute a version of the measured precision/recall curve with
         precision monotonically decreasing
    2nd) We compute the AP as the area under this curve by numerical integration.
"""
def voc_ap(rec, prec):
    """
    --- Official matlab code VOC2012---
    mrec=[0 ; rec ; 1];
    mpre=[0 ; prec ; 0];
    for i=numel(mpre)-1:-1:1
            mpre(i)=max(mpre(i),mpre(i+1));
    end
    i=find(mrec(2:end)~=mrec(1:end-1))+1;
    ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    rec.insert(0, 0.0) # insert 0.0 at begining of list
    rec.append(1.0) # insert 1.0 at end of list
    mrec = rec[:]
    prec.insert(0, 0.0) # insert 0.0 at begining of list
    prec.append(0.0) # insert 0.0 at end of list
    mpre = prec[:]
    """
     This part makes the precision monotonically decreasing
        (goes from the end to the beginning)
        matlab: for i=numel(mpre)-1:-1:1
                    mpre(i)=max(mpre(i),mpre(i+1));
    """
    # matlab indexes start in 1 but python in 0, so I have to do:
    #     range(start=(len(mpre) - 2), end=0, step=-1)
    # also the python function range excludes the end, resulting in:
    #     range(start=(len(mpre) - 2), end=-1, step=-1)
    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])
    """
     This part creates a list of indexes where the recall changes
        matlab: i=find(mrec(2:end)~=mrec(1:end-1))+1;
    """
    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1
    """
     The Average Precision (AP) is the area under the curve
        (numerical integration)
        matlab: ap=sum((mrec(i)-mrec(i-1)).*mpre(i));
    """
    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre

"""
 Draw plot using Matplotlib
"""
def draw_plot_func(dictionary, n_classes, window_title, plot_title, x_label, output_path, to_show, plot_color, true_p_bar):
    # sort the dictionary by decreasing value, into a list of tuples
    sorted_dic_by_value = sorted(dictionary.items(), key=operator.itemgetter(1))
    print(sorted_dic_by_value)
    # unpacking the list of tuples into two lists
    sorted_keys, sorted_values = zip(*sorted_dic_by_value)
    #
    if true_p_bar != "":
        """
         Special case to draw in:
            - green -> TP: True Positives (object detected and matches ground-truth)
            - red -> FP: False Positives (object detected but does not match ground-truth)
            - pink -> FN: False Negatives (object not detected but present in the ground-truth)
        """
        fp_sorted = []
        tp_sorted = []
        for key in sorted_keys:
            fp_sorted.append(dictionary[key] - true_p_bar[key])
            tp_sorted.append(true_p_bar[key])
        plt.barh(range(n_classes), fp_sorted, align='center', color='crimson', label='False Positive')
        plt.barh(range(n_classes), tp_sorted, align='center', color='forestgreen', label='True Positive', left=fp_sorted)
        # add legend
        plt.legend(loc='lower right')
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            fp_val = fp_sorted[i]
            tp_val = tp_sorted[i]
            fp_str_val = " " + str(fp_val)
            tp_str_val = fp_str_val + " " + str(tp_val)
            # trick to paint multicolor with offset:
            # first paint everything and then repaint the first number
            t = plt.text(val, i, tp_str_val, color='forestgreen', va='center', fontweight='bold')
            plt.text(val, i, fp_str_val, color='crimson', va='center', fontweight='bold')
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    else:
        plt.barh(range(n_classes), sorted_values, color=plot_color)
        """
         Write number on side of bar
        """
        fig = plt.gcf() # gcf - get current figure
        axes = plt.gca()
        r = fig.canvas.get_renderer()
        for i, val in enumerate(sorted_values):
            str_val = " " + str(val) # add a space before
            if val < 1.0:
                str_val = " {0:.2f}".format(val)
            t = plt.text(val, i, str_val, color=plot_color, va='center', fontweight='bold')
            # re-set axes to show number inside the figure
            if i == (len(sorted_values)-1): # largest bar
                adjust_axes(r, t, fig, axes)
    # set window title
    fig.canvas.set_window_title(window_title)
    # write classes in y axis
    tick_font_size = 12
    plt.yticks(range(n_classes), sorted_keys, fontsize=tick_font_size)
    """
     Re-scale height accordingly
    """
    init_height = fig.get_figheight()
    # comput the matrix height in points and inches
    dpi = fig.dpi
    height_pt = n_classes * (tick_font_size * 1.4) # 1.4 (some spacing)
    height_in = height_pt / dpi
    # compute the required figure height
    top_margin = 0.15 # in percentage of the figure height
    bottom_margin = 0.05 # in percentage of the figure height
    figure_height = height_in / (1 - top_margin - bottom_margin)
    # set new height
    if figure_height > init_height:
        fig.set_figheight(figure_height)

    # set plot title
    plt.title(plot_title, fontsize=14)
    # set axis titles
    # plt.xlabel('classes')
    plt.xlabel(x_label, fontsize='large')
    # adjust size of window
    fig.tight_layout()
    # save the plot
    fig.savefig(output_path)
    # show image
    # if to_show:
    plt.show()
    # close the plot
    # plt.close()

"""
 Plot - adjust axes
"""
def adjust_axes(r, t, fig, axes):
    # get text width for re-scaling
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    # get axis width in inches
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    # get axis limit
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1]*propotion])


def read_txt_to_list(path):
    # open txt file lines to a list
    with open(path) as f:
        content = f.readlines()
    # remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content