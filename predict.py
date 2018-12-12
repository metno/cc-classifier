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
parser.add_argument('--with_probs', type=bool, default=False, help='output probabilities')
args = parser.parse_args()


predictor = predictor.Predictor(args.modeldir, int(args.epoch))
result = predictor.predict(args.filename)

if isinstance(result, (list, tuple, np.ndarray)):
    cc_cnn = np.argmax(result[0]) # Array of probabilities
    sys.stdout.write("%d" % cc_cnn)
    if args.with_probs:
        sys.stdout.write(" [ ")
        for p in result[0]:
            sys.stdout.write("%0.2f%%, " % (p*100.0))
        sys.stdout.write(" ]")
    print("")
    #print("%d %s" % (cc_cnn, result[0]))
else:
    print(result)  # Error


