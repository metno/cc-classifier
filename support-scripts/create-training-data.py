#!/usr/bin/python3 -u

from datetime import date, timedelta
import datetime
import sqlite3
import cv2
import sys
import re
import random
import os
#from tflearn.data_utils import build_hdf5_image_dataset
#import h5py
from globals import *

# select count(*) from (select camera_id, image_timestamp, label from image_labels);
# select count(*) from (select distinct camera_id, image_timestamp, label from image_labels);


#camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2018-08-31T0605'
#camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2018-09-05T0605'

#camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2018-09-10T1250'
camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2018-09-19T0822'


conn = sqlite3.connect(camsdb)
c = conn.cursor()


image_dir = "/lustre/storeB/project/metproduction/products/webcams"
samples = dict()

    
def square_and_resize(img):
    height, width, channels = img.shape
    if width > height:  #crop image
        # crop_img = img[y:y+h, x:x+w]
        square_img = img[0:height, 0:height]
    else:               # extract from upper part of image
        square_img = img[0:width, 0:width]
    resized_img = cv2.resize(square_img, (128,128))
    return resized_img


def getlabels():
        conn = sqlite3.connect('cams.db')
        c = conn.cursor()

        dict = {}
        #count8 = 0

        for row in  c.execute('SELECT camera_id, label,image_timestamp, username FROM image_labels where  label != 9 AND status="" ORDER BY RANDOM()'):
            camid = row[0]            
            label = row[1]
                
            image_timestamp = row[2]
            username = row[3]
            
            key = str(camid) + "_" + image_timestamp; 
            if not key in dict:
                dict[key] = [];
                
            userlabel = {'camid': camid,
                         'label': label,
                         'image_timestamp': image_timestamp,
                         'username': username,
                         'label': label}
            dict[key].append(userlabel)
            
        
        labels = [];
        for k in dict:
            if len(dict[k]) > 1:
                print(dict[k])
                labels.append(random.choice(dict[k]))
            else:
                labels.append(dict[k][0])

        return labels
    
labels = getlabels()

#for row in  c.execute('SELECT distinct camera_id, label,image_timestamp FROM image_labels'):
count = 0
count8 = 0

for l in labels:
    id = l['camid']
    label = l['label']   
    image_timestamp = l['image_timestamp']
    
    myre = re.compile(r'(\d{4})(\d{2})(\d{2})T(\d{2})(\d{2})Z$')
    mo = myre.search(image_timestamp)
    if mo is not None:
        year, month, day, hour, minute = mo.groups()
    else:
        print("ERror No Match")
        continue
    image_path = ("%s/%s/%s/%s/%d/%d_%s%s%sT%s%sZ.jpg" % (image_dir, year, month, day, id, id, year, month, day, hour, minute))
    #print(image_path)
    img = cv2.imread(image_path)
    
    if img is None:
        sys.stderr.write("Corrupt image. Skipping image_path. %s\n" % image_path)
        continue

    # Skip images probably taken at night
    br = get_mean_brightness(img)
    if br < BRIGHTNESS_THRESHOLD:
        print("Skipping %s. Probably at night %d (%f)" % (image_path, label, br))
        continue
    
    image_file = ("%d_%s%s%sT%s%sZ.jpg" %(id, year, month, day, hour, minute))
    print(image_file)
    if label not in samples:
        samples[label] = []
    #samples[label].append(("training_data/%s %d" % (image_file, label)))
    
    
    try:
        resized_square_img = square_and_resize(img)
        height, width, channels = resized_square_img.shape
        cv2.imwrite('training_data/' + image_file, resized_square_img)
    except cv2.error as e:
        sys.stderr.write(e)
        continue
    count = count + 1
    #if count == 100:
    #    break
    #if label == 8:
    #    count8 = count8 + 1
    #    if count8 >= 10000:
    #        print("Skipping 8. Enough already")
    #        continue
    samples[label].append(("%s %d" % (image_file, label)))

smallest_sample = 1000000
for key, value in samples.items():
    print("%s => %d" % (key, len(value)))
    if len(value) < smallest_sample:
        smallest_sample = len(value)
        smallest_key = key
        
print("Smallest sample:  %d (cc=%s)" % (smallest_sample, smallest_key))

lines = []
for key, value in samples.items():
    #for i in range(0, smallest_sample):
    for i in range(0, len(value)):
        lines.append(value[i])

random.shuffle(lines)


#save = 1000
save = int(len(lines) / 10)
file = open("alldata.txt","w")                     
for i in range(0, len(lines) - save): # Save some for testing
    file.write(lines[i]  + os.linesep) 
file.close()

file = open("testdata.txt","w")                     
for i in range(len(lines) - save, len(lines)):
    file.write(lines[i]  + os.linesep) 
file.close()

# For hdf5 for tflearn
#file = open("trainingdata.txt","w")                     
#for i in range (int(len(lines) * 8/float(10))):  # 80% for training data ..
#    file.write(lines[i]  + os.linesep) 
#file.close()

#file = open("validationdata.txt","w")                     
#for k in range(i+1, len(lines)):  # .. And 20% for training data.
#    file.write(lines[k] + os.linesep) 
#file.close()

#print("Building hdf5 dataset ..")
# Build a HDF5 dataset for training
#build_hdf5_image_dataset('trainingdata.txt', image_shape=(128, 128), mode='file', output_path='trainingdata.h5', categorical_labels=True, normalize=True)

# Build a HDF5 dataset for validation
#build_hdf5_image_dataset('validationdata.txt', image_shape=(128, 128), mode='file', output_path='validationdata.h5', categorical_labels=True, normalize=True)

