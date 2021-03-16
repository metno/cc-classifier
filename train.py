''''exec python3 -u -- "$0" ${1+"$@"} # '''
# Using this ^ to prevent buffering running with nohup

import sys
from matplotlib import pyplot
from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from keras.preprocessing.image import ImageDataGenerator 
from keras.layers import BatchNormalization

from keras.optimizers import SGD
import dataset
import tensorflow as tf

import keras
from keras.preprocessing import image
import matplotlib.pyplot as plt
import numpy as np
leaky_relu_alpha = 0.1

def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(128, 128))
    img_tensor = image.img_to_array(img)   # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.0                                      # imshow expects values in the range [0, 1]
    
    show = False
    if show:
        plt.imshow(img_tensor[0])                           
        plt.axis('off')
        plt.show()

    return img_tensor


class predict_callback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        pass

    def on_epoch_end(self, batch, logs={}):
        new_image = load_image('/lustre/storeB/project/metproduction/products/webcams/2021/03/09/81/81_20210309T1200Z.jpg')
        pred = self.model.predict(new_image)
        cc_cnn = np.argmax(pred[0]) # Array of probabilities
        print("PRED: %d" % cc_cnn)


# load train and test dataset
def load_dataset():
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # load dataset
    trainX, trainY, testX, testY = dataset.read_train_sets2("/home/espenm/data/v52/alldata.txt",
                                                                "/home/espenm/data/v52/training_data", 
                                                                128, classes, validation_size=0.30)
    # one hot encode target values
    trainY = to_categorical(trainY, 9, dtype='float32')
    testY = to_categorical(testY, 9, dtype='float32')
    return trainX, trainY, testX, testY

# scale pixels
def prep_pixels(train, test):
    # convert from integers to floats
    train_norm = train.astype('float32')
    test_norm = test.astype('float32')
    # normalize to range 0-1
    train_norm = train_norm / 255.0
    test_norm = test_norm / 255.0
    # return normalized images
    return train_norm, test_norm

# define cnn model
def define_model():
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=leaky_relu_alpha)) 
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=leaky_relu_alpha)) 
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=leaky_relu_alpha)) 
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=leaky_relu_alpha)) 
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.3))
    
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=leaky_relu_alpha)) 
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=leaky_relu_alpha)) 
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=leaky_relu_alpha)) 
    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(LeakyReLU(alpha=leaky_relu_alpha)) 
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    
    model.add(Flatten())
    model.add(Dense(1024, activation='relu' ))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=leaky_relu_alpha)) 
    model.add(Dropout(0.5))
    model.add(Dense(9, activation='softmax'))
    # compile model
    #opt = SGD(lr=0.001, momentum=0.9, learning_rate=1e-4)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

# plot diagnostic learning curves
def summarize_diagnostics(history):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Classification Accuracy')
    pyplot.plot(history.history['accuracy'], color='blue', label='train')
    pyplot.plot(history.history['val_accuracy'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot.png')
    pyplot.close()

# run the test harness for evaluating a model
def run_test_harness():
    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                                #filepath="checkpoints/saved_model_{epoch:02d}.pb",
                                                filepath="checkpoints/saved_model_v1.pb",
                                                save_weights_only=False,
                                                verbose=0, save_freq='epoch',
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                mode='max')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="/tmp/tf")


    # This callback will stop the training when there is no improvement in
    # the validation loss for 10 consecutive epochs.
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=True, mode='auto')


    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    # prepare pixel data
    #trainX, testX = prep_pixels(trainX, testX)
    # define model
    model = define_model()
    # fit model
    
    history = model.fit(trainX, trainY, epochs=1000, batch_size=128, 
                        validation_data=(testX, testY), verbose=1,
                        callbacks=[cp_callback, tensorboard_callback])



    """
    aug = ImageDataGenerator(rotation_range=0, width_shift_range=0.0, height_shift_range=0.0, shear_range=0.1,
                                horizontal_flip=True)

    # train the network
    history = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
            validation_data=(testX, testY), steps_per_epoch=len(trainX) // 64,
            epochs=1000, callbacks=[cp_callback, tensorboard_callback], verbose=1)
    """
    # evaluate model
    _, acc = model.evaluate(testX, testY, verbose=1)
    print('> %.3f' % (acc * 100.0))
    # learning curves
    summarize_diagnostics(history)

# entry point, run the test harness
run_test_harness()
