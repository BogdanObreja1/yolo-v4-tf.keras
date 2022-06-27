import os
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import initialize_model_logger, load_image_using_tkinter, load_images, \
    predict,  output_upscaled_input_and_depth_image, calculate_distance_to_objects, segment_bounding_boxes_using_depth, \
    proposed_bounding_boxes_based_on_segmentation,load_demo_image
from models import Yolov4



# Argument Parser
parser = argparse.ArgumentParser(description='Image Depth Estimator and Object Detection')
parser.add_argument('--model', required=True, choices=['nyu.h5', 'kitti.h5'], type=str, help='Choose the trained model (nyu.h5 or kitti.h5).')
parser.add_argument('--segmentation', required=True, choices=['std', 'iqr', 'median', 'otsu'], type=str, help='Segment bounding boxes using the depth image. The algorithms that can be used for segmentation are:'
                                                                                     ' 1. Standard Deviation (L2 norm) - std;'
                                                                                     ' 2. Interquartile range - iqr;'
                                                                                     ' 3. Median thresholding - median;'
                                                                                     ' 4. Otsu Thresholding - otsu;')
parser.add_argument('--mode', required=True, choices=['demo', 'full'], type=str, help = "Allow the user to select between the following 2 modes:"
                                                                                          " 1. demo - loads a specific image as input;"
                                                                                          " 2. full - allow the user to select the image as input;")
args = parser.parse_args()

# Initializing the logger
logger_model = initialize_model_logger()

# Load the depth_model_name
depth_model_name = args.model

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

# Load depth model into GPU / CPU
depth_model = load_model(depth_model_name, custom_objects=custom_objects, compile=False)

logger_model.info('Model loaded ({0}).'.format(depth_model_name))

if args.mode == 'demo':
    # Load the the demo image.
    image = load_demo_image(logger_model, args.model)

elif args.mode == 'full':
    # Allow the user to load the input image.
    image = load_image_using_tkinter(logger_model)

    while image == ['']:
        logger_model.error("No image selected! Please try again.")
        image = load_image_using_tkinter(logger_model)

# Input images. The functions has been modified to load images of different sizes.
model_inputs, loaded_input_images_name, input_shape, org_imgs = load_images(image, logger_model, depth_model_name)
# Predict the depth image
outputs_depth_img = predict(depth_model, model_inputs, logger_model, loaded_input_images_name)

# Returns the upscaled depth images normalized and the input image
# Writes in the output folder the input and the depth image.
input_image, depth_img, depth_img_normalized = output_upscaled_input_and_depth_image(outputs_depth_img.copy(), logger_model, input_shape, org_imgs.copy(), loaded_input_images_name)

# Import the yolo_model
yolo_model = Yolov4(weight_path='yolov4.weights',
               class_name_path='class_names/coco_classes.txt')


# Predict objects in the input img using the YOLO model.
detections = yolo_model.predict(input_image, logger_model, plot_img=False)

# Calculates the distances from the camera to the object in meters.
detections_with_distances = calculate_distance_to_objects(depth_img_normalized, detections, depth_model_name,logger_model)

# Output the YOLO image (including the distances to objects) in the output_images folder.
yolo_image = yolo_model.output_yolo_image_with_distances(input_image,logger_model,loaded_input_images_name[0], detections_with_distances)

# Load the segmentation algorithm
segmentation_algorithm = args.segmentation

# Segment bounding boxes using the depth image.
segmented_depth_images = segment_bounding_boxes_using_depth(input_image, depth_img_normalized, detections_with_distances, logger_model,loaded_input_images_name[0], segmentation_algorithm)

# New proposed boxes    Save the segmented box image in the output_images folder
new_proposed_detections = proposed_bounding_boxes_based_on_segmentation(segmented_depth_images, detections_with_distances)

# Output the new YOLO image
new_yolo_image = yolo_model.output_yolo_image_with_distances(input_image,logger_model,loaded_input_images_name[0], new_proposed_detections, True, segmentation_algorithm)

