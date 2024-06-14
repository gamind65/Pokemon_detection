
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2
import torchvision.transforms as transforms 
import streamlit as sl
import torchvision
import numpy as np
import cv2 as cv
import random
import torch

# define a transform funtion for FasterRCNN
transform = transforms.Compose([ 
    transforms.ToTensor() 
])

class model():
    def __init__(self, model_path, classes):
        # load FasterRCNN architechture, number of classes and classes
        self.backbone = fasterrcnn_resnet50_fpn_v2(pretrained=False)
        self.classes = classes
        
        # load pretrained weights
        features = self.backbone.roi_heads.box_predictor.cls_score.in_features
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        
        # load weights into backbone
        self.backbone.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(features, len(self.classes))
        self.backbone.load_state_dict(checkpoint.state_dict(), strict=False)
        self.backbone.eval()
            
        
    def show(self, img, boxes, labels, scores):
        # iterates through results
        for bbox, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = [int(coord) for coord in bbox] # get x1, y1, x2, y2 (left, top, right, bottom)
            
            # draw bounding box, write name and confidence scores
            color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) # get random color
            cv.rectangle(img, (x1, y1), (x2, y2), color, 8) # draw box
            cv.putText(img, f'{self.classes[int(label)-1]}-{"%.2f"%score}', (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 1, color, 4)  # write name and scores
            
        # show image
        sl.image(np.asarray(img), channels="BGR", width=750)
            
    def detect(self, image, threshold):
        sl.write('Detecting image...')
        # preprocess input image
        img_transform = transform(image).unsqueeze(0) 
        
        # detect
        output = self.backbone(img_transform)

        # get results informations
        idx = output[0]['scores'] > threshold # get index of instances that confidence pass threshold
        filtered_boxes = output[0]['boxes'][idx] # get boxes corr
        filtered_labels = output[0]['labels'][idx] # get labels 
        filtered_scores = output[0]['scores'][idx] # get confidence scores
        
        if len(filtered_boxes) < 0:
            return
        
        else:
            # get name of labels
            labels_out = [self.classes[int(i)-1] for i in filtered_labels]
            
            # show image with results
            self.show(image, filtered_boxes, filtered_labels, filtered_scores)
            sl.write(labels_out)
            
            return list(set(labels_out))
        