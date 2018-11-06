#!/usr/bin/python3 -u

from datetime import date, timedelta
import datetime
import sqlite3
import cv2
import sys
import re
import random
import os
import astral
from globals import *

import pytz
utc = pytz.utc

# From A. Mariano, MacOS units(1), 1993.
FT_PER_METRE = 3.2808399


# v28
#camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2018-09-20T0605'

# v29
#camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2018-10-03T0605'


# V30
#camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2018-10-12T0605'

# v31 small
#camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2018-10-25T0605'

# v32
camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2018-11-06T0605'

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
    conn = sqlite3.connect(camsdb)
    c = conn.cursor()
    c2 = conn.cursor()

    dict = {}
    #count8 = 0

    for row in  c.execute('SELECT il.camera_id, il.label, il.image_timestamp, il.username, wc.latitude, wc.longitude FROM image_labels il, webcams wc where  il.label != 9 AND wc.id=il.camera_id AND wc.status = "" ORDER BY RANDOM()'):

        camid = row[0]
        label = row[1]
        image_timestamp = row[2]
        username = row[3]
        lat = row[4]
        lon = row[5]
        key = str(camid) + "_" + image_timestamp;

        if not key in dict:
            dict[key] = [];

        userlabel = {'camid': camid,
                     'label': label,
                     'image_timestamp': image_timestamp,
                     'username': username,
                     'label': label,
                     'lat': lat,
                     'lon': lon
        }
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
    lat = float(l['lat'])
    lon = float(l['lon'])

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

    image_file = ("%d_%s%s%sT%s%sZ.jpg" %(id, year, month, day, hour, minute))
    #print(image_file)
    
    timestamp1 = datetime.datetime.strptime( image_timestamp, "%Y%m%dT%H%MZ" )
    #print(timestamp1)
    #print("Lat: %f, Lon: %f" %(lat/100.0, lon/100.0))
    loc = astral.Location(info=("",
                            "",
                            lat/100.0,
                            lon/100.0,
                            "UTC",
                            157/FT_PER_METRE))

    try:
        result = loc.sun(date=timestamp1)
        #for k in ["dawn", "sunrise", "noon", "sunset", "dusk"]:
        #    print("%7s %s" % (k, result[k].astimezone(utc).replace(tzinfo=None)))

        dusk = result['dusk'].astimezone(utc).replace(tzinfo=None)
        dawn = result['dawn'].astimezone(utc).replace(tzinfo=None)

        if timestamp1  >  dusk + datetime.timedelta(minutes=10):
            print("%s is later than then minutes before dusk (%s). Skipping" % (image_file, dusk))
            continue

        if timestamp1  <  dawn + datetime.timedelta(minutes=10):
            print("%s is earlier than then minutes before dawn (%s). Skipping" % (image_file, dawn))
            continue
        
        print("%s, Dusk: %s, Dawn: %s" % (image_file, dusk, dawn) )
    except astral.AstralError as e:
        if str(e) == "Sun never reaches 6 degrees below the horizon, at this location.":
            print("Midnight sun: %s" % image_file)
        elif str(e) == "Sun never reaches the horizon on this day, at this location.":
            print("Mørketid:  s" % image_fille)
            continue
        else:
            print("%s %s" % (image_file, e))
    except Exception as e2:
        print("%s %s" % (image_file, e2))
        continue

    # Skip images probably taken at night
    br = get_mean_brightness(img)
    if br < BRIGHTNESS_THRESHOLD: # Should mostly not reach here due to the dusk/dawn testing above
        print("Skipping %s. Probably at night %d (%f)" % (image_path, label, br))
        continue

    
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
