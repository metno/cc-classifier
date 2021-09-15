#!/usr/bin/env python3

import os, sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import argparse
import re
import numpy as np
from pandas import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

parser = argparse.ArgumentParser(description='Create a confusio namtrix')
parser.add_argument('--predictions', type=str, help=" A file with lines containine path ccobs ccpred")
args = parser.parse_args()

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

cccnt = [0, 0, 0, 0, 0, 0, 0, 0, 0]

with open(args.predictions, "r") as ins:
    for line in ins:
        myre = re.compile(r'(\S+)\s+(\d) (-?\d)$')
        mo = myre.search(line.strip())
        if mo is not None:
            image_file, cc, cc_cnn = mo.groups()
            cc = int(cc)
            cc_cnn = int(cc_cnn)
            cccnt[cc] = cccnt[cc] + 1 
        else:
            print("ERror No Match")
            continue

        #if abs(cc_cnn - cc) <= 2:
        #    confusion_matrix[cc][cc] = confusion_matrix[cc][cc] + 1
        #else:
        confusion_matrix[cc_cnn][cc] = confusion_matrix[cc_cnn][cc] + 1
#print(cccnt.index(min(cccnt)))
#print(min(cccnt))

cccnt2 = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
count = 0
with open(args.predictions, "r") as ins:
    for line in ins:
        myre = re.compile(r'(\S+)\s+(\d) (-?\d)$')
        mo = myre.search(line.strip())
        if mo is not None:
            image_file, cc, cc_cnn = mo.groups()
            cc = int(cc)
            cc_cnn = int(cc_cnn)
            cccnt2[cc] = cccnt2[cc] + 1
            if cccnt2[cc] > min(cccnt):
                continue
            count = count + 1
        else:
            print("ERror No Match")
            continue

        #if abs(cc_cnn - cc) <= 2:
        #    confusion_matrix[cc][cc] = confusion_matrix[cc][cc] + 1
        #else:
        #confusion_matrix[cc_cnn][cc] = confusion_matrix[cc_cnn][cc] + 1
        #if cc_cnn != cc:
        #    print("%d %d"  %(cc, cc_cnn))

df_cm = DataFrame(confusion_matrix)
print(df_cm)
plt.figure(figsize = (10,7))
title = 'Confusion matrix'

print(title)
fig, ax = plt.subplots()
plt.title(title)
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,fmt="d", annot_kws={"size": 12})# font size
plt.savefig('confusion.png', dpi=400)
