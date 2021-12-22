from collections import OrderedDict


class Config:
  def __init__(self):
    self.__dict__ = OrderedDict([
      ('weights', './checkpoints/yolov4-416'), # path to weights file
      ('classes', './data/classes/coco.names'), # path to classes
      ('model', 'yolov4'), # yolov3 or yolov4
      ('size', 416), # resize images to
      ('tiny', False), # yolo or yolo-tiny
      ('framework', 'tf'), # tf, tflite, trt,
      ('iou', 0.45), # iou threshold
      ('score', 0.50), # score threshold
      ('count', False), # count objects within images
      ('info', True), # print info on detections
      ('crop', False), # crop detections from images
      ('output', './detections/'), # path to output folder
      ('plate', False), # perform license plate recognition
      ('ocr', False) # perform generic OCR on detection regions
    ])


config = Config()
