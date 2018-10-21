'''
In this script, we provide a simple yet powerful function that uses
image augmentation techniques.
This helps to deal with less number of training instances,
increase accuracy, generalize models, etc.

Mandatory packages to be installed:
    tensorflow
    numpy
'''

import tensorflow as tf
import numpy as np


def augment_data2(dataset, dataset_labels, label_counts):

    counts = label_counts.copy()
    augmented_images = []
    augmented_image_labels = []

    use_random_rotation = False
    use_random_shift = False   # This is no good ## Not enough RAM
    use_random_shear = False
    use_copy = True
    
    num_augs_enabled = 0
    if use_random_rotation:
        num_augs_enabled = num_augs_enabled + 1
    if use_random_shift:
        num_augs_enabled = num_augs_enabled + 1
    if use_random_shear:
        num_augs_enabled = num_augs_enabled + 1

    print("Num augs enabled: %d" % num_augs_enabled)
    aug_factors = dict()
    for ccval in range(0, 9):  # cloud coverage, values in [0,8]
        if num_augs_enabled == 0:
            continue
        aug_factors[ccval] = round((label_counts[8]/num_augs_enabled) / label_counts[ccval])
    print(aug_factors)
    """
    print("dataset.load_training_data(): label %d, "
    "Aug_factor: %f, "
    "Num images: %f, "
    "Num images after oversampling: %f" %
    (ccval,
    aug_factors[ccval],
    label_counts[ccval],
    aug_factors[ccval] * label_counts[ccval] * num_augs_enabled))
    """

    #maximg = {0: 3000, 1: 3000, 2: 3000, 3: 3000, 4: 3000, 5: 3000, 6: 3000, 7: 0, 8: 0}

    # v28 # localhost
    

    # Enough RAM on Floydhub
    #maximg = {0: 8000, 1: 8000, 2: 8000, 3: 8000, 4: 12000, 5: 12000, 6: 12000, 7: 12000, 8: 0}
    maximg = {0: 8000, 1: 0, 8000: 8000, 3: 8000, 4: 8000, 5: 8000, 6: 8000, 7: 8000, 8: 0}
    #maximg = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
    # Enough RAM on neo
    #maximg = {0: 7000, 1: 5000, 2: 6000, 3: 6000, 4: 7000, 5: 7000, 6: 7000, 7: 7000, 8: 0}
    imgcounts = {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0, 8:0}
    for num in range (0, dataset.shape[0]):
        if num % 1000 == 0:
            print("Augmenting %d .." % num)

        cc = dataset_labels[num].tolist().index(1.0)
        if counts[cc] >= maximg[cc]:
            continue

        counts[cc] = counts[cc] + 1

        for i in range(0, aug_factors[cc] + 1) :
            if use_copy:
                augmented_images.append(dataset[num])
            

            if use_random_rotation is True:
                augmented_images.append(tf.contrib.keras.preprocessing.image.random_rotation(dataset[num],
                                                                                                                                                                         5,
                                                                                                                                                                         row_axis=0,
                                                                                                                                                                         col_axis=1,
                                                                                                                                                                         channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])
                counts[cc] = counts[cc] + 1

            if use_random_shear is True:
                augmented_images.append(tf.contrib.keras.preprocessing.image.random_shear(dataset[num],
                                                                                                                                                                  0.15,
                                                                                                                                                                  row_axis=0,
                                                                                                                                                                  col_axis=1,
                                                                                                                                                                  channel_axis=2))
                augmented_image_labels.append(dataset_labels[num])
                counts[cc] = counts[cc] + 1



            if use_random_shift is True:
                augmented_images.append(tf.contrib.keras.preprocessing.image.random_shift(dataset[num],
                                                                                                                                                                  0.15,
                                                                                                                                                                  0.0,
                                                                                                                                                                  row_axis=0,
                                                                                                                                                                  col_axis=1,
                                                                                                                                                                  channel_axis=2,
                                                                                                                                                                  fill_mode='wrap'))
                augmented_image_labels.append(dataset_labels[num])
                counts[cc] = counts[cc] + 1


    print("Training imgcounts: ")
    print(imgcounts)
    print("Training images after augmentation: ")
    print(counts)
    return np.array(augmented_images), np.array(augmented_image_labels)
