import os
import argparse



os.environ['TF_CPP_MIN_LOG_LEVEL'] = '5'
from keras.models import load_model
from layers import BilinearUpSampling2D
from utils import initialize_model_logger, select_pretrained_model_for_inference, load_image_using_tkinter, load_images, \
    predict,  output_upscaled_input_and_depth_image, calculate_distance_to_objects
from matplotlib import pyplot as plt
from models import Yolov4


# Initializing the logger
logger_model = initialize_model_logger()

# UI that allows the user to select the depth pre-trained model on which to run inference.
depth_model_name = select_pretrained_model_for_inference(logger_model)

# Argument Parser
parser = argparse.ArgumentParser(description='High Quality Monocular Depth Estimation via Transfer Learning')
parser.add_argument('--model', default=depth_model_name, type=str, help='Trained Keras model file.')
args = parser.parse_args()

# Custom object needed for inference and training
custom_objects = {'BilinearUpSampling2D': BilinearUpSampling2D, 'depth_loss_function': None}

# Load model into GPU / CPU
depth_model = load_model(args.model, custom_objects=custom_objects, compile=False)

logger_model.info('Model loaded ({0}).'.format(args.model))

# Allow the user to load the input images
image = load_image_using_tkinter(logger_model)

while image == ['']:
    logger_model.error("No image selected! Please try again.")
    image = load_image_using_tkinter(logger_model)

# Input images. The functions has been modified to load images of different sizes.
# Check load_images in utils.py
inputs, loaded_input_images_name = load_images(image, logger_model)

# Predict the depth image
outputs_depth_img = predict(depth_model, inputs, logger_model, loaded_input_images_name)

# Returns the upscaled depth images normalized/not-normalized and the input image
# Writes in the output folder the input and the depth image.
input_image, depth_img, depth_img_normalized = output_upscaled_input_and_depth_image(outputs_depth_img.copy(), logger_model, inputs.copy(), loaded_input_images_name)

# Import the yolo_model
yolo_model = Yolov4(weight_path='yolov4.weights',
               class_name_path='class_names/coco_classes.txt')


# Predict objects in the input img using the YOLO model.
detections = yolo_model.predict(input_image, logger_model, plot_img=False)

#print(yolo_image.shape)


#test_images = calculate_distance_from_object(depth_img_normalized,yolo_image, detections,depth_model_name, logger_model)

# Calculates the distances from the camera to the object in meters.
detections_with_distances = calculate_distance_to_objects(depth_img_normalized, detections, depth_model_name,logger_model)

# Output the YOLO image (including the distances to objects) in the output_images folder.
yolo_image = yolo_model.output_yolo_image_with_distances(input_image,logger_model,loaded_input_images_name[0], detections_with_distances)

#improve_segm
