from ultralytics import YOLO
import streamlit as sl
from PIL import Image
import numpy as np

class model():
    def __init__(self, file_path):
        self.backbone = YOLO(file_path)
    
    def detect(self, image, threshold):
        # detect
        sl.write('Detecting image...')
        res = self.backbone.predict(image, conf=threshold)
        results = res[0]
                    
        # get and draw boxes
        for r in res:
            im_array = r.plot()
            im = Image.fromarray(im_array[..., ::-1])
        im = np.array(im)
        sl.image(im, width=750)
        
        # get names 
        names = []
        for result in results:
            detection_count = result.boxes.shape[0]
            for i in range(detection_count):
                cls = int(result.boxes.cls[i].item())
                names.append(result.names[cls])
        
        # get pokemons information
        if len(names) > 0:
            sl.write('Detected pokemons:', names)
        names = [new.lower() for new in list(set(names))]
        
        return names