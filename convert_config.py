from collections import OrderedDict


class Config:
  def __init__(self):
     self.__dict__ = OrderedDict([
        ('weights', './data/yolov4.weights'), # path to weights file
        ('classes', './data/classes/coco.names'), # path to classes
        ('output', './checkpoints/yolov4-416'), # path to output
        ('tiny', False), # is yolo-tiny or not
        ('input_size', 416), # define input size of export model
        ('score_thres', 0.2), # define score threshold
        ('framework', 'tf'), # define what framework do you want to convert (tf, trt, tflite)
        ('model', 'yolov4') # yolov3 or yolov4
    ])


config = Config()
