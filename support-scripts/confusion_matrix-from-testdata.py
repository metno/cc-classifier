#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import re
import predictor
import numpy as np
from pandas import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

parser = argparse.ArgumentParser(description='Create a confusio namtrix')
parser.add_argument('--labelsfile', type=str, help="A labels file containing lines like this:\n fileNNN.jpg 6")
parser.add_argument('--modeldir', type=str, help='Model dir', default='modeldata')
parser.add_argument('--epoch', type=str, help='epoch', default=888)
args = parser.parse_args()

predictor = predictor.Predictor(args.modeldir, int(args.epoch))

imagedir = '/lustre/storeB/project/metproduction/products/webcams'

confusion_matrix = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0 ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0 ]
]


count = 0
cc2labels = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: []}

with open(args.labelsfile, "r") as ins:
    for line in ins:
        myre = re.compile(r'(\S+)\s+(-?\d)$')
        mo = myre.search(line.strip())
        if mo is not None:
            image_file, cc = mo.groups()
            cc = int(cc)
        else:
            print("ERror No Match")
            continue
        myre = re.compile(r'(\d+)_(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})Z.jpg$')
        mo = myre.search(image_file)
        if mo is not None:
            id, year, month, day, hour, minute = mo.groups()
        else:
            print("Error No Match")
            continue

        path = ("%s/%s/%s/%s/%s/%s" %(imagedir, year, month, day, id, image_file ) )
        result = predictor.predict(path)
        if isinstance(result, (list, tuple, np.ndarray)):
            cc_cnn = np.argmax(result[0]) # Array of probabilities
            print("%s %s %d " % (path, cc, cc_cnn))
        else:
            cc_cnn = result  # Error

        if abs(cc_cnn - cc) <= 2:
            confusion_matrix[cc][cc] = confusion_matrix[cc][cc] + 1
        else:
            confusion_matrix[cc_cnn][cc] = confusion_matrix[cc_cnn][cc] + 1

        count = count + 1
        if count == 1000:
            break

#print(np.matrix(confusion_matrix))
df_cm = DataFrame(confusion_matrix)
print(df_cm)
plt.figure(figsize = (10,7))
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,annot_kws={"size": 16})# font size
plt.savefig('confusion.png', dpi=400)
