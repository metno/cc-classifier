''''exec python3 -u -- "$0" ${1+"$@"} # '''
# Using this ^ to prevent stdout buffering running with nohup

import sys
from matplotlib import pyplot
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

validation_split = 0.25
# load train and test dataset
def load_dataset():
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    # load dataset
    trainX, trainY, testX, testY = dataset.read_train_sets2("/home/espenm/data/v54_half/alldata.txt",
                                                                "/home/espenm/data/v54_half/training_data", 
                                                                128, classes, validation_size=validation_split)
    # one hot encode target values
    trainY = to_categorical(trainY, 9, dtype='float32')
    testY = to_categorical(testY, 9, dtype='float32')
    return trainX, trainY, testX, testY

# Define cnn model
def define_model():

    # intput 128    
    model = Sequential()
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    # This time it seems "Drop out" actually is the chance of 
    # "dropping out", and not the odds of "staying". Keras vs Tensorflow
    model.add(Dropout(0.6))
    
    # 256 
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.7))
    
    # 512
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.8))
    
    # 1024
    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(1024, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))    
    model.add(Dropout(0.7))

    # 2048
    #model.add(Conv2D(2048, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(Conv2D(2048, (3, 3), activation='relu', padding='same'))
    #model.add(BatchNormalization())
    #model.add(MaxPooling2D((2, 2)))
    #model.add(Dropout(0.5))
    
    # Output
    model.add(Flatten())
    model.add(Dense(1024, activation='relu' ))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Dense(9, activation='softmax'))
    # compile model
    opt = SGD(lr=0.001, momentum=0.9, learning_rate=1e-3)
    #opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    [print(n.name) for n in tf.compat.v1.get_default_graph().as_graph_def().node]

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

# train model
def train():
    # Create a callback that saves the model + model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                                filepath="checkpoints/saved_model_{epoch:03d}.pb",
                                                #filepath="checkpoints/saved_model_v1.pb",
                                                save_weights_only=False,
                                                verbose=0, save_freq='epoch',
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                mode='max')

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="/tmp/tf")


    # This callback will stop the training when there is no improvement in
    # the validation loss for 100 consecutive epochs.
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=125, verbose=True, mode='auto')


    # load dataset
    trainX, trainY, testX, testY = load_dataset()
    
    # define model
    model = define_model()
    # fit model
    
    history = model.fit(trainX, trainY, epochs=1000, batch_size=128,
                        validation_data=(testX, testY), verbose=1,
                        callbacks=[cp_callback, tensorboard_callback, early_stopping_callback])


    """
    aug = ImageDataGenerator(rotation_range=0, width_shift_range=0.0, height_shift_range=0.0, shear_range=0.1,
                                horizontal_flip=True)

    # train the network
    history = model.fit_generator(aug.flow(trainX, trainY, batch_size=64),
            validation_data=(testX, testY), steps_per_epoch=len(trainX) // 64,
            epochs=1000, callbacks=[cp_callback, tensorboard_callback], verbose=1)
    """
    # evaluate model
    scores = model.evaluate(testX, testY, verbose=1)
    print(model.metrics_names)
    print(scores)
    # learning curves
    summarize_diagnostics(history)

# entry point, train model
if __name__ == "__main__":
    train()
