#!/usr/bin/env python3
# load_model_sample.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
logging.getLogger('tensorflow').disabled = True
from keras.models import load_model
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np

import argparse
#import cv2

parser = argparse.ArgumentParser(description='Do cloud coverage preditcion on image')
parser.add_argument('--filename', type=str, help='Input image to do prediction on')
parser.add_argument('--modelpath', type=str, help='Input image to do prediction on')
args = parser.parse_args()

def load_image(img_path, show=False):

    #img = cv2.imread(img_path)
    #img = cv2.resize(img, (128, 128),0,0, cv2.INTER_LINEAR)

    img = image.load_img(img_path, target_size=(128, 128))
    img_tensor = image.img_to_array(img)   # (height, width, channels)
    img_tensor = np.array(img, dtype=np.float32)
    img_tensor = np.expand_dims(img_tensor, axis=0)  # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.0                              # imshow expects values in the range [0, 1]
   
    show = False
    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor


if __name__ == '__main__':
    

    # load model
    model = load_model(args.modelpath)
    
    # image path
    #img_path = '/lustre/storeB/project/metproduction/products/webcams/2021/03/09/81/81_20210309T1200Z.jpg'
    #img_path = '/lustre/storeB/project/metproduction/products/webcams/2021/03/09/82/82_20210309T1200Z.jpg' 
    #img_path = '/lustre/storeB/project/metproduction/products/webcams/2021/03/09/142/142_20210309T1200Z.jpg'
    # load a single image
    new_image = load_image(args.filename)

    # check prediction
    #pred = model.predict(new_image)
    pred = model.predict(new_image)
    cc_cnn = np.argmax(pred[0]) # Array of probabilities
    #cc_cnn = np.argmax(pred, axis=1)
    #cc_cnn = pred.argmax(axis=1)[0]
    #print(pred)
    print(cc_cnn)
