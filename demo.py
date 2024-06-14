import requests
import cv2 as cv
import numpy as np 
from PIL import Image
import streamlit as sl

from utils import SSD_model
from utils import YOLO_model
from utils import PokemonInfo
from utils import FasterRCNN_model

#------------------------------------------------------------------------------------------------------------------------------#
# set up streamlit interface layout
sl.set_page_config(layout="wide")

# load custom labels for FasterRCNN-ResNet50v1-fpn
fast_classes = ['cinderace', 'dracovish', 'dragonite', 'eevee', 'eternatus', 'gengar', 'grookey', 'inteleon',
           'lucario', 'meowth', 'mew', 'mr-mime', 'morpeko', 'pikachu', 'sirfetchd', 'wobbuffet', 'yamper',
           'zacian', 'zamazenta']

@sl.cache_resource()
def loadPretrainedModel():
    # load pretrained yolo model
    yolo_model = YOLO_model.model('./Models/yolo_model.pt')
    
    # load pretrained FasterRCNN model
    fast_model = FasterRCNN_model.model('./Models/model-epoch200.pt', fast_classes)
    
    # load SSD model
    ssd_dir = './Models/saved_model'
    labels_path = './Models/label_map.txt'
    ssd_model = SSD_model.model(ssd_dir, labels_path, fast_classes)
    
    return yolo_model, fast_model, ssd_model 
    
yolo_model, fast_model, ssd_model = loadPretrainedModel()

#------------------------------------------------------------------------------------------------------------------------------#
def init_col1(col1):
    """This function defines the intitial appearance of column in the left in streamlit interface"""
    
    # configurating column 1
    with col1:
        # selection box for model selection
        sl.header("Model")
        model_select = sl.selectbox('Choose your detection model',
                              ('YOLO-V8x','FasterRCNN-ResNet50', 'SSD-ResNet50'),
                              index = None,
                              placeholder='Select your model...',)
        
        sl.write('You selected', model_select)
        
        sl.write('---')
        
        # slider for threshold adjustment
        sl.header("Threshold")
        thres_slider = sl.slider('Adjust detection threshold',
                            0.0, 1.0, 0.5, 0.05)
        
        sl.write('You selected', thres_slider)
        
        # classify buttons
        detect_button = sl.button("Detect", type="primary")
        
        return model_select, thres_slider, detect_button 
    
def init_col2(col2):
    """This function defines the intitial appearance of column in the right in streamlit interface"""
    
    # configurating column 2
    with col2:
        # Introduction
        sl.header("Pokemon detection")
        
        # Image selecting
        sl.subheader("Choose your image...")
        uploaded_file = None
        uploaded_file = sl.file_uploader("Choose a file...")
        if uploaded_file is not None:
            sl.image(uploaded_file, width=750)
            
            # convert uploaded file to image
            pil_image = Image.open(uploaded_file).convert("RGB")
            img_array = np.array(pil_image)
            img_array = cv.cvtColor(img_array, cv.COLOR_BGR2RGB)
            return img_array
        else:
            sl.write('Please upload your image')
            return None

#----------------------------------------------------------------------------------------------------------------------------    
def detect_image(col, model, threshold, button, image):
    """This function doing detecting work of image
    Arguments:
    - col: column to show result
    - model_select: chosen model to do detection work
    - threshold: threshold for detection
    - button: streamlit button, signal for start detect
    - image: uploaded image for detection
    """

    with col:
        if button:
            if (model is None) or (image is None):
                sl.write('Please upload image or select detection model')
            else:
                detected_names = None
                if model == 'YOLO-V8x':
                    detected_names = yolo_model.detect(image, threshold)
                if model == 'FasterRCNN-ResNet50':
                    detected_names = fast_model.detect(image, threshold)
                if model == 'SSD-ResNet50':
                    detected_names = ssd_model.detect(image, threshold)
                    
                if (detected_names == None) or (len(detected_names) == 0):
                    sl.write('No pokemon detected!')
                    return
            
                for name in detected_names:
                    if name == 'mimey':
                        name = 'mr-mime'
                    elif name == "sirfetch'd":
                        name = 'sirfetchd'
                    elif name == 'morpeko':
                        name = 'morpeko-full-belly' 
                    sl.markdown("""---""")
                    sl.write('Pokemon name:', name)
                    
                    request = requests.get(f'https://pokeapi.co/api/v2/pokemon/{name}')
                    PokemonInfo.get_pokemon_info(request.json())
    
        
#-------------------------------------------------------------------------------------------#
def main():
    # initiate 2 columns in streamlit UI
    col1, col2 = sl.columns([0.3, 0.7])
    
    # create the two columns in streamlit interface
    model_select, thres_slider, detect_button = init_col1(col1)
    uploaded_image = init_col2(col2)
    
    # do detection
    detect_image(col2, model_select, thres_slider, detect_button, uploaded_image)


if __name__ == '__main__':
    main()    