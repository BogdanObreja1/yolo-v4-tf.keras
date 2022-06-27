import unittest
import os

class TestPretrainedModels(unittest.TestCase):

    def test_models(self):
        """
        Check if the pretrained models ('nyu.h5','kitti.h5', yolo4.weights) are in the project folder.
        """
        list_files = os.listdir()
        kitty_file_name = 'kitti.h5'
        nyu_file_name = 'nyu.h5'
        yolo_file_name = 'yolov4.weights'
        self.assertIn(kitty_file_name,list_files, "The pretrained Kitti model h5 file had not been found in the project!")
        self.assertIn(nyu_file_name,list_files, "The pretrained NYU model h5 file had not been found in the project!")
        self.assertIn(yolo_file_name, list_files,"The pretrained YOLO model weights had not been found in the project!")


if __name__ == '__main__':
    unittest.main()