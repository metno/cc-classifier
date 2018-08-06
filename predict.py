#!/usr/bin/python3 -u

import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import math
import predictor

filename =sys.argv[1] 

cpdir = './cc-predictor-model'

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
checkpoint = 623


predictor = predictor.Predictor(cpdir, checkpoint)
result = predictor.predict(filename)

if isinstance(result, (list, tuple, np.ndarray)):
    cc_cnn = np.argmax(result[0]) # Array of probabilities
    #print(result[0])
else:
    cc_cnn = result  # Error

print(cc_cnn)
