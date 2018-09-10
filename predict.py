#!/usr/bin/python3 -u

import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse
import math
import predictor
import argparse




#cpdir = './cc-predictor-model'
#cpdir = './modeldata'

parser = argparse.ArgumentParser(description='Do cloud coverage preditcion on image')
parser.add_argument('--filename', type=str, help='Input image to do prediction on')
parser.add_argument('--modeldir', type=str, help='Model dir', default='modeldata')
parser.add_argument('--epoch', type=str, help='epoch', default=888)
args = parser.parse_args()


predictor = predictor.Predictor(args.modeldir, int(args.epoch))
result = predictor.predict(args.filename)

if isinstance(result, (list, tuple, np.ndarray)):
    cc_cnn = np.argmax(result[0]) # Array of probabilities
    #print(result[0])
else:
    cc_cnn = result  # Error

print(cc_cnn)
