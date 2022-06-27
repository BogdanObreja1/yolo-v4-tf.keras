import unittest
import os
from utils import initialize_model_logger, resize_input_image, convert_to_rgb_format
import numpy as np
from PIL import Image



class TestUtils(unittest.TestCase):

    def setUp(self):

        # Initializing the logger (needed for testing functions)
        self.logger_model = initialize_model_logger()

        dir = os.path.dirname(__file__)
        # Directory for unittest images
        self.image1_path = os.path.join(dir, 'images_unittest', "img1.png")
        self.image2_path = os.path.join(dir, 'images_unittest', "img2.png")
        self.image3_path = os.path.join(dir, 'images_unittest', "img3.png")
        self.image4_path = os.path.join(dir, 'images_unittest', "img4.png")

    def test_resize_input_image(self):
        """
        Check if the resized input image shape is the same as the test/training image.
        """
        image1 = np.clip(np.asarray(Image.open(self.image1_path), dtype=float)/255, 0, 1)
        image1_resized, _ = resize_input_image(image1, self.logger_model, "img1.png", "nyu.h5")
        image4 = np.clip(np.asarray(Image.open(self.image4_path), dtype=float) / 255, 0, 1)
        image4_resized, _ = resize_input_image(image4, self.logger_model, "img4.png", "kitti.h5")

        self.assertEqual([image1_resized.shape[0], image1_resized.shape[1]],[480,640], "Image dimensions don't match!")
        self.assertEqual([image4_resized.shape[0], image4_resized.shape[1]], [384,1280],"Image dimensions don't match!")

    def test_RGB_conversion(self):
        """
        Check if the loaded images that are not RGB format (with 3 channels) are correctly converted to the model requirement.
        """

        image2 = np.array(Image.open(self.image2_path))
        image3 = np.array(Image.open(self.image3_path))

        image2 = convert_to_rgb_format(image2, self.image2_path, self.logger_model, "img2.png")
        image3 = convert_to_rgb_format(image3, self.image3_path, self.logger_model, "img3.png")

        self.assertEqual([image2.ndim,image2.shape[-1]], [3, 3], "Unable to convert the img2.png to RGB format")
        self.assertEqual([image3.ndim, image3.shape[-1]], [3, 3], "Unable to convert the img3.png to RGB format")










#if __name__ == '__main__':
#    unittest.main()
