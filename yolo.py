import os

# comment out below line to enable tensorflow outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    
import cv2
import numpy as np
import yolo_core.utils as utils
from tensorflow.python.saved_model import tag_constants
from yolo_core.functions import *
from yolo_core.yolov4 import filter_boxes

from model_config import config


class YoloModel:
    def __init__(self, conf=config):
        self.loaded = False
        self.config = conf
        self.load_model()

    def set_config(self, config):
        self.config = config
        self.load_model()

    def load_model(self):
        weights_exists = os.path.exists(self.config.weights)
        classes_exists = os.path.exists(self.config.classes)

        if weights_exists and classes_exists:
            if self.config.framework == 'tflite':
                self.interpreter = tf.lite.Interpreter(model_path=self.config.weights)
                self.loaded = True
            else:
                self.saved_model_loaded = tf.saved_model.load(self.config.weights, tags=[tag_constants.SERVING])
                self.loaded = True

    def predict(self, original_image, allowed_classes=[]):
        if not self.loaded:
            raise Exception("Model not loaded!")

        input_size = self.config.size
        
        if not isinstance(allowed_classes, list):
            allowed_classes = []

        if isinstance(original_image, str):
            original_image = cv2.imread(original_image)
            
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        image_data = cv2.resize(original_image, (input_size, input_size))
        image_data = image_data / 255.

        images_data = []
        for i in range(1):
            images_data.append(image_data)
        images_data = np.asarray(images_data).astype(np.float32)

        if self.config.framework == 'tflite':
            self.interpreter.allocate_tensors()
            input_details = self.interpreter.get_input_details()
            output_details = self.interpreter.get_output_details()
            self.interpreter.set_tensor(input_details[0]['index'], images_data)
            self.interpreter.invoke()
            pred = [self.interpreter.get_tensor(output_details[i]['index']) for i in range(len(output_details))]

            if self.config.model == 'yolov3' and self.config.tiny == True:
                boxes, pred_conf = filter_boxes(pred[1], pred[0], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
            else:
                boxes, pred_conf = filter_boxes(pred[0], pred[1], score_threshold=0.25, input_shape=tf.constant([input_size, input_size]))
        else:
            infer = self.saved_model_loaded.signatures['serving_default']
            batch_data = tf.constant(images_data)
            pred_bbox = infer(batch_data)
            for key, value in pred_bbox.items():
                boxes = value[:, :, 0:4]
                pred_conf = value[:, :, 4:]

        # run non max suppression on detections
        boxes, scores, classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(boxes, (tf.shape(boxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                pred_conf, (tf.shape(pred_conf)[0], -1, tf.shape(pred_conf)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=self.config.iou,
            score_threshold=self.config.score
        )

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = original_image.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)

        # format bounding boxes from normalized ymin, xmin, ymax, xmax ---> xmin, ymin, xmax, ymax
        original_h, original_w, _ = original_image.shape
        bboxes = utils.format_boxes(boxes.numpy()[0], original_h, original_w)
        
        # hold all detection data in one variable
        detection = [bboxes, scores.numpy()[0], classes.numpy()[0], valid_detections.numpy()[0]]
        
        # format and return predictions
        return self.format_output(detection, allowed_classes)

    def format_output(self, detection, pred_allowed_classes):
        predictions = []
        classes = utils.read_class_names(self.config.classes)
        num_classes = len(classes)

        allowed_classes = list(classes.values())

        if len(pred_allowed_classes):
            allowed_classes = pred_allowed_classes

        out_boxes, out_scores, out_classes, num_boxes = detection

        for i in range(num_boxes):
            if int(out_classes[i]) < 0 or int(out_classes[i]) > num_classes: continue
            coor = out_boxes[i]
            score = out_scores[i]
            class_ind = int(out_classes[i])
            class_name = classes[class_ind]

            if class_name not in allowed_classes:
                continue

            predictions.append({
                "class": class_name,
                "confidence": score,
                "xmin": coor[0], 
                "ymin": coor[1], 
                "xmax": coor[2], 
                "ymax": coor[3]
            })


        return predictions

model = YoloModel()
