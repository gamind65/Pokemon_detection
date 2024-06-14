import torchvision.transforms.functional as TF
import tensorflow as tf
import streamlit as sl
import numpy as np
import cv2 as cv
import random

# Load the custom labels from .txt file for SSD
def load_custom_labels(labels_path):
    with open(labels_path, 'r') as file:
        lines = file.readlines()
    category_index = {}
    for i, line in enumerate(lines):
        category_index[i + 1] = line.strip()
    return category_index

class model():
    def __init__(self, model_path, label_path, class_name):
        self.backbone = tf.saved_model.load(model_path)
        self.classes = load_custom_labels(label_path)
        self.class_name = class_name
        
    def detect(self, image, threshold):
        sl.write('Detecting image...')
        
        # preprocess input image
        input_tensor = tf.convert_to_tensor(image)
        input_tensor = input_tensor[tf.newaxis,...]

        # detect
        detections = self.backbone(input_tensor)
        class_name =  self.show(detections, self.classes, image, threshold)
        return class_name
    
    def show(self, detections, image_np, threshold):
        detection_boxes = detections['detection_boxes'][0].numpy()
        detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
        detection_scores = detections['detection_scores'][0].numpy()

        image_with_detections = image_np.copy()
        class_name = []
        
        for i in range(len(detection_boxes)):
            if detection_scores[i] >= threshold:
                box = detection_boxes[i]
                class_sample = self.classes[detection_classes[i]]
                for name in self.class_name:
                    if name in class_sample.lower():
                        class_name.append(name)
                        
                score = detection_scores[i]
                ymin, xmin, ymax, xmax = box
                (left, right, top, bottom) = (xmin * image_np.shape[1], xmax * image_np.shape[1],
                                            ymin * image_np.shape[0], ymax * image_np.shape[0])
                
                color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
                image_with_detections = cv.rectangle(image_with_detections, (int(left), int(top)), (int(right), int(bottom)), color, 4)
                image_with_detections = cv.putText(image_with_detections, f'{class_name[-1]}-{"%.2f"%score}', (int(left), int(top) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, color, 4)

        if len(class_name) < 1:
            return
        
        else:
            sl.image(image_with_detections, channels="BGR", width=750)
            sl.write('Detected pokemons:', class_name)
            return class_name
