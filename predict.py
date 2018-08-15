#!/usr/bin/python3 -u

import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import math
import predictor
import argparse



filename =sys.argv[1] 
#cpdir = './cc-predictor-model'
cpdir = './modeldata'

#parser = argparse.ArgumentParser(description='Do cloud coverage preditcion on image')
#parser.add_argument('--filename', type=str, help='Input image to do prediction on')
#parser.add_argument('--modeldir', type=str, help='Model dir')
#parser.add_argument('--epoch', type=str, help='ecpoch')
#args = parser.parse_args()

## Iteration 397131 Training Epoch 1316 --- Training Accuracy: 100.0%, Validation Accuracy: 100.0%,  Validation Loss: 0.022
#checkpoint = 4541

# v5
#checkpoint = 4129
# v6
#checkpoint = 433

# v7
#checkpoint = 3682

# v8
#checkpoint = 3060

# v11
#checkpoint = 1428

# v11-python3-rotate-augmentation .. 
#checkpoint = 623

checkpoint = 326

predictor = predictor.Predictor(cpdir, checkpoint)
result = predictor.predict(filename)

if isinstance(result, (list, tuple, np.ndarray)):
    cc_cnn = np.argmax(result[0]) # Array of probabilities
    #print(result[0])
else:
    cc_cnn = result  # Error

print(cc_cnn)
