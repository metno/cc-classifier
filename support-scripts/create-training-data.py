#!/bin/sh
''''exec /usr/bin/env python3 -u -- "$0" ${1+"$@"} # '''
#!/usr/bin/env python3

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

# v41
#camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2019-01-28T1501'

# v46
#camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2019-03-21T0605'

# v47
# camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2019-03-28T0605'

# v48 
# camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2019-04-08T0605'

# v49
#camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2019-04-24T0605'

# V50
#camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2019-04-29T0605'

# v51
#camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2019-08-07T0605'

# v52
#camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2019-09-16T0605'


# v53 - With low brightness
#camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2019-10-03T0605'

# v54 - Take back brighness test
#camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db.2019-10-03T0605'

# v55 - Take back brighness test
camsdb = '/lustre/storeB/project/metproduction/static_data/camsatrec/cams.db'


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

    #for row in  c.execute('SELECT il.camera_id, il.label, il.image_timestamp, il.username, wc.latitude, wc.longitude FROM image_labels il, webcams wc where  il.label != 9 AND wc.id=il.camera_id AND wc.status = "" ORDER BY RANDOM()'):
    for row in  c.execute("SELECT il.camera_id, il.label, il.image_timestamp, il.username, wc.latitude, wc.longitude FROM image_labels il, webcams wc where  il.label != 9 AND wc.id=il.camera_id AND wc.status not like '%obstructed%' ORDER BY RANDOM()"):

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

    
    #if image_timestamp != '20180213T1600Z':
    #    continue
    
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
    br = get_mean_brightness(img)
    # Skip images probably taken at night    
    if br < BRIGHTNESS_THRESHOLD: # Should mostly not reach here due to the dusk/dawn testing above
        print("Skipping %s. Probably at night %d (%f)" % (image_path, label, br))
        continue
    
    timestamp1 = datetime.datetime.strptime( image_timestamp, "%Y%m%dT%H%MZ" )
    timestamp2 = datetime.datetime.strptime( image_timestamp, "%Y%m%dT%H%MZ" )
    timestamp2 = timestamp2.replace(minute=0, hour=12, second=0, microsecond=0)

    
    loc = astral.Location(info=("",
                            "",
                            lat/100.0,
                            lon/100.0,
                            "UTC",
                            157/FT_PER_METRE))

    try:
        result = loc.sun(date=timestamp2)
        #for k in ["dawn", "sunrise", "noon", "sunset", "dusk"]:
        #    print("%7s %s" % (k, result[k].astimezone(utc).replace(tzinfo=None)))

        # Skumring
        dusk = result['dusk'].astimezone(utc).replace(tzinfo=None)
        if timestamp1  >  dusk - datetime.timedelta(minutes=35):
            print("%s is later than 35 minutes before dusk (%s). Skipping" % (image_file, dusk))
            continue
        
        # Morgengry
        dawn = result['dawn'].astimezone(utc).replace(tzinfo=None)
        if timestamp1  <  dawn + datetime.timedelta(minutes=35):
            print("%s is earlier than 35 minutes after dawn (%s). Skipping" % (image_file, dawn))
            continue
        
        print("%s, Dusk: %s, Dawn: %s" % (image_file, dusk, dawn) )
    except astral.AstralError as e:
        if str(e) == "Sun never reaches 6 degrees below the horizon, at this location.":
            print("Midnight sun, Sun never reaches 6 degrees below the horizon, at this location.: %s (%f %f br: %f)" % (image_file, lat/100.0, lon/100.0, br))
        elif str(e) == "Sun never reaches the horizon on this day, at this location.":
            print("Mørketid:  %s" % image_file)
            continue
        else:
            print("%s %s" % (image_file, e))
    except Exception as e2:
        print("%s %s" % (image_file, e2))
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
smallest_key = -1
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
