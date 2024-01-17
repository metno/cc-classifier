#!/usr/bin/env python3

import os, sys
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


parser = argparse.ArgumentParser(description='Create a confusio namtrix')
parser.add_argument('--predictions', type=str, help="A file with lines containing \"^path answer(int) prediction(int)\"")
args = parser.parse_args()
dims = 3

confusion_matrix = [[0] * dims for _ in range(dims)]
count = 0

with open(args.predictions, "r") as ins:
    for line in ins:
        myre = re.compile(r'^(\S+)\s(\d)\s(-?\d)')
        mo = myre.search(line.strip())
        if mo is not None:
            image_file, cc, cc_cnn = mo.groups()
            cc = int(cc)
            cc_cnn = int(cc_cnn)
          
        else:
            print("ERror No Match1")
            continue

        confusion_matrix[cc_cnn][cc] = confusion_matrix[cc_cnn][cc] + 1


df_cm = pd.DataFrame(confusion_matrix)
print(df_cm)
plt.figure(figsize = (10,7))
title = 'Confusion matrix'

print(title)
fig, ax = plt.subplots()
plt.title(title)
sn.set(font_scale=1.4)#for label size
sn.heatmap(df_cm, annot=True,fmt="d", annot_kws={"size": 12})# font size
plt.savefig('confusion.png', dpi=400)
