#!/usr/bin/env python3

#!/bin/sh
''''exec python -u -- "$0" ${1+"$@"} # '''
# vi: syntax=python



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


def calc_spread(vector):
    i = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]) / 8
    x = np.array(vector)
    mean = np.sum(x * i)
    variance = sum(i * i * x) - mean*mean
    return variance

if isinstance(result, (list, tuple, np.ndarray)):
    cc_cnn = np.argmax(result[0]) # Array of probabilities
    sys.stdout.write("%d" % cc_cnn)
    if args.with_probs:
        sys.stdout.write(" probabilities: [ ")
        for p in result[0]:
            sys.stdout.write("%0.2f%%, " % (p*100.0))
        sys.stdout.write(" ]")
        sys.stdout.write(" spread: %.02f" % calc_spread(result[0]))
    print("")
    #print("%d %s" % (cc_cnn, result[0]))
else:
    print(result)  # Error


