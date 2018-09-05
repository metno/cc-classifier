#!/usr/bin/python3 -u
import re
import cv2
import predictor
import numpy as np
#import matplotlib.pyplot as plt

cpdir = 'modeldata/v24_4500'

#v21
checkpoint = 1634

predictor = predictor.Predictor(cpdir, checkpoint)

imagedir = '/lustre/storeB/project/metproduction/products/webcams'

cnt = 0
x = []
y1 = []
y2 = []

def calc_spread(vector):
    i = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]) / 8
    x = np.array(vector)
    mean = np.sum(x * i)
    variance = sum(i * i * x) - mean*mean
    return variance

with open("testdata.txt", "r") as ins:
    for line in ins:
        myre = re.compile(r'(\S+)\s+(-?\d)$')
        mo = myre.search(line.strip())
        if mo is not None:
            image_file, cc = mo.groups()
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
            cc_dnn = np.argmax(result[0]) # Array of probabilities
            print("%s %s %d %f" % (path, cc, cc_dnn, calc_spread(result[0])))
        else:
            cc_dnn = result  # Error
                  

        
        cnt = cnt  + 1
        #if cnt  == 50:
        #    break
        #try:            
        #    image = cv2.imread(path)
        #except cv2.error as e:
        #    print(e)
        #    continue

#plt.plot(x, y1, color='green', linewidth=1)
#plt.plot(x, y2, color='blue', linewidth=1)
#plt.show()
print("Done!")
