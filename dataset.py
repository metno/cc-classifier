

# This script was initially from a cv-tricks.com tutorial
# It has MIT licence


import re
import cv2
import numpy as np
import os
from sklearn.utils import shuffle
import tensorflow as tf
import sys

import augment

class DataSet(object):

  def __init__(self, images, labels):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    #self._img_names = img_names
    #self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

  @property
  def labels(self):
    return self._labels

  #@property
  #def img_names(self):
  #  return self._img_names

  #@property
  #def cls(self):
  #  return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images[start:end], self._labels[start:end]


def load_training_data(labelsfile, imagedir, image_size, classes):
    images = []
    labels = []
    #img_names = []
    #cls = []
    
    print("Loading ..")
    cnt = 0;
    with open(labelsfile, "r") as ins:
        for line in ins:
            cnt = cnt  + 1
            #if cnt % 600  == 0:
            #    print("loaded %d" % cnt)
            #    break
            myre = re.compile(r'(\S+)\s+(-?\d)$')
            mo = myre.search(line.strip())
            if mo is not None:
                path, cc = mo.groups()
            else:
                print("Error: No match")
                continue
            try:
                image = cv2.imread(imagedir + "/" + path)
            except cv2.error as e:
                print(e)
                continue

            if image is None:
                print("image %s is none" % path)
                continue
            #Already resized
            #image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            if int(cc) < 0:
                continue
            
            index = classes.index(int(cc))
            image = image.astype(np.float32)
            # convert from [0:255] => [0.0:1.0]
            image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)            
            
    images = np.array(images)
    labels = np.array(labels)

    
    aug_images, aug_labels = augment.augment_data(images, labels,
                                                  use_random_rotation=True,
                                                  #use_random_shift=True , # This is no good ## Not enough RAM
                                                  use_random_shear=True, # Not enough RAM  
                                                  use_random_zoom=False,
						  skip_labels = [],       # Skip augment label 8.
						  augementation_factor = 1) # Of times to run the  
	                                                # (random) augmentation
    images = np.concatenate([images, aug_images])
    labels = np.concatenate([labels, aug_labels])
    

    return images, labels

def read_train_sets(labelsfile, imagedir, image_size, classes, validation_size):
  class DataSets(object):
    pass
  data_sets = DataSets()

 
  images, labels = load_training_data(labelsfile, imagedir, image_size, classes)
  print("SIZE: %d" % (sys.getsizeof(images) / (1024*1024)))
    
  images, labels = shuffle(images, labels)  

    
  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  #validation_img_names = img_names[:validation_size]
  #validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  #train_img_names = img_names[validation_size:]
  #train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels)
  data_sets.valid = DataSet(validation_images, validation_labels)

  return data_sets

# Test            
if __name__ == "__main__":
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    #load_training_data("alldata.txt", 128, classes)
    #load_training_data("alldata.txt", 128, classes)
    data = read_train_sets("alldata.txt", 128, classes, 0.30)
