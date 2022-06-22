from models import Yolov4

model = Yolov4(weight_path='yolov4.weights',
               class_name_path='class_names/coco_classes.txt')
img = model.predict('people.png')