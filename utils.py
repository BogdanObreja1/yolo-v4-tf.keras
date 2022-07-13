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

def load_demo_image(logger_model, model_name):
    """
    Load the demo image.
    """

    dir = os.path.dirname(__file__)
    # Output directory for images
    input_dir = os.path.join(dir, 'input_images', 'demo_images')
    if model_name == "nyu.h5":
        image_path = input_dir + "//people.png"

    elif model_name == "kitti.h5":
        image_path = input_dir +"//000296.png"

    logger_model.info("Loade Image: " + str(image_path))

    return [image_path]



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


def load_images(image_files, logger_model, model_name):
    """"
    (Created function)
    -> Added the option to load images of different sizes (before all input images had to be the same size).
    -> Makes sure to convert non-RGB format images to the RGB standard format (3 channels) - Requirement for the model.
      (Palettised colored images can also be used)
    -> Resize the input image(s) to the image sizes used during training/testing (nyu - (640, 480), kitti - (1280, 384)).
       The resulting image(s) will be used as input for depth inference.
    -> Also returns the input image with original size after converting to RGB format in order to be used later as input
     for YOLO.
    """
    loaded_images = []
    loaded_images_name = []
    org_images = []
    for file in image_files:
        x = np.array(Image.open(file))
        parsed_file_name = file.split("/")
        input_image_name = parsed_file_name[-1]
        loaded_images_name.append(input_image_name)
        x = convert_to_rgb_format(x, file, logger_model, input_image_name)
        input_image = copy.deepcopy(x)
        org_images.append(np.stack(input_image))
        x, input_shape = resize_input_image(x, logger_model, input_image_name, model_name)
        loaded_images.append(np.stack(x))

    # return np.stack(loaded_images, axis=0)
    return loaded_images, loaded_images_name, input_shape, org_images


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


def resize_input_image(image, logger_model, input_image_name, pretrained_model):
    """"
    Resize the input image(s) to the image sizes used during training/testing (nyu - 640, 480, kitti - 1280, 384).
    The resulting image(s) will be used as input for depth inference.
    The encoder architecture also expects the image dimensions to be divisible by 32.
    Resizing method used: bicubic (to retain as many details as possible)
    (Created Function)
    """
    height , width, channels = image.shape
    input_shape = (height, width)
    is_input_image_upscaled = 0

    if pretrained_model == "nyu.h5":
        output_shape = (480, 640)
    elif pretrained_model == 'kitti.h5':
        output_shape = (384, 1280)

    if width % output_shape[1] != 0:
        image = resize(image, output_shape, order=3, preserve_range=True, mode='reflect', anti_aliasing=True)
        is_input_image_upscaled = 1

    if height % output_shape[0] != 0:
        image = resize(image, output_shape, order=3, preserve_range=True, mode='reflect', anti_aliasing=True)
        is_input_image_upscaled = 1

    if is_input_image_upscaled == 1:
        logger_model.info(
            input_image_name + " has been resized from " + str(input_shape) + " to " + str(output_shape) +
            " in order to be used as input for the depth inference.")

    return image,input_shape


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


def output_upscaled_input_and_depth_image(outputs, logger_model,input_shape, inputs=None, loaded_input_images_name=None ):
    """
    (Created Function)
    Returns the upscaled depth images normalized and the input image
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
        #upscaled_depth_image_shape = (2 * outputs[index][0].shape[0], 2 * outputs[index][0].shape[1])
        upscaled_depth_image_shape = input_shape
        upscaled_depth_image_normalized = to_multichannel(resize(outputs[index][0], upscaled_depth_image_shape, order=3, mode='reflect', anti_aliasing=True))
        upscaled_depth_image_normalized = upscaled_depth_image_normalized[:,:,0]
        upscaled_depth_image = np.uint8(to_multichannel(resize(outputs[index][0], upscaled_depth_image_shape, order=3, mode='reflect', anti_aliasing=True)) * 255)
        upscaled_depth_image = upscaled_depth_image[:, :, 0]
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
    The distance from camera to the object is calculated using the median of the depth image ROI.
    In order to find distances in meters we must multiply the result from the previous step by the maximum distance of the training dataset (10meters for NYU, 80meters for KITTI).
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
    (Created Function)
    Calculates the distance to object in meters using median.
    """

    # Median Depth Image ROI
    depth_median_normalized = np.median(depth_img_roi)
    # Value converted in meters
    depth_median_in_meters = depth_median_normalized * range_model
    logger_model.info("The distance to Object_" + cls + " _ID_" + str(index) + " using median is: " + str(depth_median_in_meters) + " meters.")

    return depth_median_in_meters


def segment_bounding_boxes_using_depth(rgb_input_image, depth_img_normalized, detections, logger_model, input_image_name, segmentation_mode, cut_off_gain = 'default'):
    """
    (Created Function)
    Segment bounding boxes using the depth image. The algorithms that can be used for segmentation are:
    1. Standard Deviation (L2 norm) - "std"
    2. Interquartile range - "iqr"
    3. Median thresholding - "median"
    4. Otsu Thresholding - "otsu"
    5. Automatic Canny ("auto-canny") - was not a robust method (hence was removed)
    """
    # Segment each individual object box detected
    output_depth_segmentation_images = []
    for index, row in detections.iterrows():
        x1, y1, x2, y2, cls, score, w, h, distances = row.values
        depth_img_roi_normalized = copy.deepcopy(depth_img_normalized[y1:y2, x1:x2])
        rgb_img_roi = copy.deepcopy(rgb_input_image[y1:y2, x1:x2])
        output_segmented_box_img = None
        output_segmented_depth_img = None
        if segmentation_mode == "std":
            output_segmented_box_img, output_segmented_depth_img = segment_box_img_using_std(depth_img_roi_normalized, rgb_img_roi, index, cls, logger_model, cut_off_gain)

        elif segmentation_mode == "iqr":
            output_segmented_box_img, output_segmented_depth_img = segment_box_img_using_iqr(depth_img_roi_normalized, rgb_img_roi, index, cls, logger_model, cut_off_gain)

        elif segmentation_mode == "median":
            output_segmented_box_img, output_segmented_depth_img = segment_box_img_using_median(depth_img_roi_normalized, rgb_img_roi, index, cls,
                                                                 logger_model, cut_off_gain)

        elif segmentation_mode == "otsu":
            output_segmented_box_img, output_segmented_depth_img = segment_box_img_using_otsu(depth_img_roi_normalized, rgb_img_roi, index, cls,
                                                                    logger_model)
        """
        elif segmentation_mode == "auto-canny":
            output_segmented_box_img = segment_box_img_using_auto_canny(depth_img_roi_normalized, rgb_img_roi, index, cls,
                                                                  logger_model, 0.3)
        """

        save_output_segmented_box_image(output_segmented_box_img,index, logger_model, input_image_name, segmentation_mode)

        output_depth_segmentation_images.append(output_segmented_depth_img)

    return output_depth_segmentation_images

def segment_box_img_using_std(depth_img_roi_normalized, rgb_img_roi, index, cls, logger_model, cut_off_gain = 'default'):
    """
    Segment the bounding box ROI using a Standard Deviation cutoff.
    """

    # Mean Depth Image ROI
    depth_mean_normalized = np.mean(depth_img_roi_normalized)

    # Standard Deviation ROI
    depth_l2_normalized = np.std(depth_img_roi_normalized)

    if cut_off_gain == "default":
        cut_off_gain = 1

    cut_off_std = depth_l2_normalized * cut_off_gain
    lower_std, upper_std = depth_mean_normalized - cut_off_std, depth_mean_normalized + cut_off_std

    rgb_img_roi[depth_img_roi_normalized < lower_std] = 0
    rgb_img_roi[depth_img_roi_normalized > upper_std] = 0

    depth_img_roi_normalized[depth_img_roi_normalized < lower_std] = 0
    depth_img_roi_normalized[depth_img_roi_normalized > upper_std] = 0

    logger_model.info("Object_" + cls + " _ID_" + str(index) + " box had been segmented using STD with a cutoff gain of " + str(cut_off_gain) + " !")

    return rgb_img_roi, depth_img_roi_normalized

def segment_box_img_using_iqr(depth_img_roi_normalized, rgb_img_roi, index, cls, logger_model, cut_off_gain = 'default'):
    """
    Segment the bounding box ROI using a Interquartile range (IQR) cutoff.
    """

    q75, q25 = np.percentile(depth_img_roi_normalized, [75, 25])
    iqr = q75 - q25

    if cut_off_gain == "default":
        cut_off_gain = 0.5

    cut_off_iqr = iqr * cut_off_gain

    lower_iqr, upper_iqr = q25 - cut_off_iqr, q75 + cut_off_iqr

    rgb_img_roi[depth_img_roi_normalized < lower_iqr] = 0
    rgb_img_roi[depth_img_roi_normalized > upper_iqr] = 0

    depth_img_roi_normalized[depth_img_roi_normalized < lower_iqr] = 0
    depth_img_roi_normalized[depth_img_roi_normalized > upper_iqr] = 0

    logger_model.info(
        "Object_" + cls + " _ID_" + str(index) + " box had been segmented using IQR with a cutoff gain of " + str(
            cut_off_gain) + " !")

    return rgb_img_roi, depth_img_roi_normalized


def segment_box_img_using_median(depth_img_roi_normalized, rgb_img_roi, index, cls, logger_model, cut_off_gain = 'default'):
    """
    Segment the bounding box ROI using median thresholding.
    """

    # Median Depth Image ROI
    depth_median_normalized = np.median(depth_img_roi_normalized)

    if cut_off_gain == "default":
        cut_off_gain = 0.2

    cut_off = depth_median_normalized * cut_off_gain
    lower_cutoff, upper_cutoff = depth_median_normalized - cut_off, depth_median_normalized + cut_off

    rgb_img_roi[depth_img_roi_normalized < lower_cutoff] = 0
    rgb_img_roi[depth_img_roi_normalized > upper_cutoff] = 0

    depth_img_roi_normalized[depth_img_roi_normalized < lower_cutoff] = 0
    depth_img_roi_normalized[depth_img_roi_normalized > upper_cutoff] = 0

    logger_model.info("Object_" + cls + " _ID_" + str(index) + " box had been segmented using median thresholding with a cutoff threshold of " + str(cut_off_gain) + " !")

    return rgb_img_roi, depth_img_roi_normalized


def segment_box_img_using_otsu(depth_img_roi_normalized, rgb_img_roi, index, cls, logger_model):
    """
    Segment the bounding box ROI using otsu thresholding.
    """
    # Convert pixel values from 0...1 to 0...255
    depth_img_roi = np.uint8(copy.deepcopy(depth_img_roi_normalized) * 255)

    # Apply binary and otsu thresholding
    ret, otsu_threshold = cv2.threshold(depth_img_roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    rgb_img_roi[otsu_threshold == 0] = 0

    logger_model.info("Object_" + cls + " _ID_" + str(index) + " box had been segmented using Otsu thresholding !")

    return rgb_img_roi, otsu_threshold

def segment_box_img_using_auto_canny(depth_img_roi_normalized, rgb_img_roi, index, cls, logger_model, median_threshold):
    """
    Segment the bounding box ROI using automatic canny.
    I had given up on it since the only way I was able to make it work decently was putting the canny edges range
    manually. The automatic canny didn't work well.
    """
    # Convert pixel values from 0...1 to 0...255
    depth_img_roi = np.uint8(copy.deepcopy(depth_img_roi_normalized) * 255)

    median_depth = np.median(depth_img_roi)
    lower = int(max(0, (1.0 - median_threshold) * median_depth))
    upper = int(min(255, (1.0 + median_threshold) * median_depth))
    # 10, 20 works well with the people.png image.
    edges_img = cv2.Canny(depth_img_roi, lower, upper)

    ret, th2 = cv2.threshold(edges_img, 100, 255, cv2.THRESH_BINARY)


    logger_model.info("Object_" + cls + " _ID_" + str(index) + " box had been segmented using automatic Canny !")

    cv2.imshow("img", th2)
    cv2.waitKey(0)


def save_output_segmented_box_image(output_segmented_box_image, index, logger_model, input_image_name, segmentation_mode):
    """
    Save the output segmented object in the output_images folder.
    """

    dir = os.path.dirname(__file__)
    # Output directory for images
    output_dir = os.path.join(dir, 'output_images')

    input_image_name = input_image_name.split(".")
    name_pic = input_image_name[0]
    segmentation_box_name = name_pic + "_yolo_z_segmentation_" + segmentation_mode + "_" + str(index) + ".png"
    full_path = output_dir + "\\" + segmentation_box_name
    im = Image.fromarray(output_segmented_box_image)
    im.save(full_path)
    logger_model.info(segmentation_box_name + " finished processing! Image can be found in the output_images folder.")

def proposed_bounding_boxes_based_on_segmentation(segmented_box_images, detections_with_distances):
    """
    The new proposed bounding boxes based on the segmented box images.
    First we find the non-zero outer points (x1,x2,y1,y2).
    Then we modify the dataframe "detections_with_distances" with the new bounding boxes values.
    """
    y1_values = []
    y2_values = []
    x1_values = []
    x2_values = []
    for depth_segmented_roi in segmented_box_images:
        outer_points = np.nonzero(depth_segmented_roi)
        y1 = outer_points[0].min()
        y2 = outer_points[0].max()
        x1 = outer_points[1].min()
        x2 = outer_points[1].max()
        y1_values.append(y1)
        y2_values.append(y2)
        x1_values.append(x1)
        x2_values.append(x2)

    detections_with_distances["y2"] = detections_with_distances["y2"] - (
                detections_with_distances["y2"] - y2_values - detections_with_distances["y1"]) + 1
    detections_with_distances["y1"] = detections_with_distances["y1"] + y1_values

    detections_with_distances["x2"] = detections_with_distances["x2"] - (detections_with_distances["x2"] - x2_values - detections_with_distances["x1"]) + 1
    detections_with_distances["x1"] = detections_with_distances["x1"] + x1_values
    detections_with_distances["w"] = detections_with_distances["x2"] - detections_with_distances["x1"]
    detections_with_distances["h"] = detections_with_distances["y2"] - detections_with_distances["y1"]

    return detections_with_distances





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