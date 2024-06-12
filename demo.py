import random
import torch
import requests
import cv2 as cv
import torchvision
import numpy as np 
from PIL import Image
from io import BytesIO
import streamlit as sl
import tensorflow as tf
from ultralytics import YOLO
import torchvision.transforms as transforms 
import torchvision.transforms.functional as TF
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, fasterrcnn_resnet50_fpn_v2

#------------------------------------------------------------------------------------------------------------------------------#
# set up streamlit interface layout
sl.set_page_config(layout="wide")

# Load the custom labels from .txt file for SSD
def load_custom_labels(labels_path):
    with open(labels_path, 'r') as file:
        lines = file.readlines()
    category_index = {}
    for i, line in enumerate(lines):
        category_index[i + 1] = line.strip()
    return category_index

@sl.cache_resource()
def loadPretrainedModel():
    # load pretrained yolo model
    #yolo_file = os.path.join(os.path.dirname(__file__), "Models\yolo_model.pt")
    yolo_model = YOLO('./Models/yolo_model.pt')
    
    # load pretrained FasterRCNN model
    fast_model = fasterrcnn_resnet50_fpn_v2(pretrained=False)
    num_classes = 19
    in_features = fast_model.roi_heads.box_predictor.cls_score.in_features
    fast_model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    checkpoint = torch.load("./Models/model-epoch200.pt", map_location=torch.device('cpu'))
    fast_model.load_state_dict(checkpoint.state_dict(), strict=False)
    fast_model.eval()
    
    # load SSD model
    ssd_dir = './Models/saved_model'
    ssd_model = tf.saved_model.load(ssd_dir)
    
    return yolo_model, fast_model, ssd_model 
    
yolo_model, fast_model, ssd_model = loadPretrainedModel()

# load custom labels for FasterRCNN-ResNet50v1-fpn
CLASSES = ['cinderace', 'dracovish', 'dragonite', 'eevee', 'eternatus', 'gengar', 'grookey', 'inteleon',
           'lucario', 'meowth', 'mew', 'mr-mime', 'morpeko', 'pikachu', 'sirfetchd', 'wobbuffet', 'yamper',
           'zacian', 'zamazenta']

# define a transform funtion for FasterRCNN
transform = transforms.Compose([ 
    transforms.ToTensor() 
]) 
labels_path = './Models/label_map.txt'
category_index = load_custom_labels(labels_path)

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
        
#---------------------------------------------------------------------------------------------------------------------------------------------#
# find pokemon avatar 
def get_pokemon_index(data):
    return data['id']

def get_pokemon_avatar(data):
    return data['sprites']['other']['official-artwork']['front_default']

# get pokemon element
def get_pokemon_element(data):
    return [data['types'][i]['type']['name'] for i in range(len(data['types']))]

# get pokemon evolution chain
def get_pokemon_evo_chain(data):
    # get evolution_chain json data
    try:
        evo_data = requests.get(requests.get(data['species']['url']).json()['evolution_chain']['url']).json()
        evo_name = []
        # get name of this pokemon based, first and second evolution
        evo_name.extend((evo_data['chain']['species']['name'],
                        evo_data['chain']['evolves_to'][0]['species']['name'],
                        *[evo_data['chain']['evolves_to'][0]['evolves_to'][i]['species']['name'] for i in range(len(evo_data['chain']['evolves_to'][0]['evolves_to']))]))
        return ' -> '.join(name for name in evo_name)
    
    except:
        return 'No evolution'

def get_pokemon_info(data):
    """Show"""
    sl.write(f'Pokedex number: #{get_pokemon_index(data)}')
    
    # show images
    img = Image.open(BytesIO(requests.get(get_pokemon_avatar(data)).content))
    sl.image(img, channels='BGR', width=350)
    
    # show element(s)
    sl.write('Element:', ', '.join(get_pokemon_element(data)))
    
    # show evolution chain
    sl.write('\nEvolution Chain:', get_pokemon_evo_chain(data))

#-----------------------------------------------------------------------------------------------------------------------------------------------#
def yolo_detect(image, threshold):
    # detect
    sl.write('Detecting image...')
    res = yolo_model.predict(image, conf=threshold)
    results = res[0]
    names = []
                
    # get and draw boxes
    for r in res:
        im_array = r.plot()
        im = Image.fromarray(im_array[..., ::-1])
    im = np.array(im)
    sl.image(im, width=750)
    
    # get names 
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


def show(img, boxes, labels, scores):
    for bbox, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        cv.rectangle(img, (x1, y1), (x2, y2), color, 8)
        cv.putText(img, f'{CLASSES[int(label)-1]}-{"%.2f"%score}', (x1, y1-10), cv.FONT_HERSHEY_SIMPLEX, 1, color, 4)
    sl.image(np.asarray(img), channels="BGR", width=750)
        
def faster_detect(image, threshold):
    sl.write('Detecting image...')
    img_transform = transform(image).unsqueeze(0) 
    output = fast_model(img_transform)

    idx = output[0]['scores'] > threshold
    filtered_boxes = output[0]['boxes'][idx]
    filtered_labels = output[0]['labels'][idx] 
    filtered_scores = output[0]['scores'][idx]
    
    if len(filtered_boxes) < 0:
        #sl.write('No pokemon detected')
        return
    else:
        labels_out = [CLASSES[int(i)-1] for i in filtered_labels]
        labels = [f"{CLASSES[i]}: {score:.2f}" for i, score in zip(filtered_labels, filtered_scores)]
        #result = draw_bounding_boxes(torch.tensor(image).permute(2,0,1), boxes=filtered_boxes, width=8, labels=labels)
        show(image, filtered_boxes, filtered_labels, filtered_scores)
        sl.write(labels_out)
        
        return list(set(labels_out))
        

def ssd_detect(image, threshold):
    sl.write('Detecting image...')
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis,...]

    # Perform detection
    detections = ssd_model(input_tensor)
    class_name =  postprocess_detections(detections, category_index, image, threshold)
    return class_name

# Postprocess the detection results
def postprocess_detections(detections, category_index, image_np, threshold):
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(np.int32)
    detection_scores = detections['detection_scores'][0].numpy()

    image_with_detections = image_np.copy()
    class_name = []
    
    for i in range(len(detection_boxes)):
        if detection_scores[i] >= threshold:  # You can adjust the threshold
            box = detection_boxes[i]
            class_sample = category_index[detection_classes[i]]
            for name in CLASSES:
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
        return None
    else:
        sl.image(image_with_detections, channels="BGR", width=750)
        sl.write('Detected pokemons:', class_name)
        return class_name

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
                    detected_names = yolo_detect(image, threshold)
                if model == 'FasterRCNN-ResNet50':
                    detected_names = faster_detect(image, threshold)
                if model == 'SSD-ResNet50':
                    detected_names = ssd_detect(image, threshold)
                    
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
                    get_pokemon_info(request.json())
    
        
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