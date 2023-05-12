#!/usr/bin/env python3 

# Importing the models to try transfer learning on
from tensorflow.keras.applications.resnet50 import ResNet50
#from tensorflow.keras.applications.resnet50_v2 import ResNet50V2
#from tensorflow.keras.applications.resnet50 import DenseNet201

#from keras.applications.resnet50 import ResNet50
from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from sklearn.utils import compute_class_weight
import numpy as np 
import keras
from collections import Counter
import tensorflow
from keras.preprocessing import image
# Setup some callbacs 
cp_callback = tensorflow.keras.callbacks.ModelCheckpoint(
                                                filepath="checkpoints/saved_model_{epoch:03d}.pb",
                                                #filepath="checkpoints/saved_model_v1.pb",
                                                save_weights_only=False,
                                                verbose=0, save_freq='epoch',
                                                monitor='val_accuracy',
                                                save_best_only=True,
                                                mode='max')

tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir="./tensorboard-data")

# This callback will stop the training when there is no improvement in
# the validation loss for 100 consecutive epochs.
early_stopping_callback = tensorflow.keras.callbacks.EarlyStopping(monitor='val_loss', patience=125, verbose=True, mode='auto')




# */ // LATEST :-) !! */
train_data_dir='/lustre/storeB/users/espenm/data/v2.0.3/train+validation/'
img_size = 128
batch_size = 128



def preprocess(img):
    width, height = img.shape[0], img.shape[1]
    img = image.array_to_img(img, scale=False)

    # Crop 48x48px
    desired_width, desired_height = 48, 48

    if width < desired_width:
        desired_width = width
    start_x = np.maximum(0, int((width-desired_width)/2))

    img = img.crop((start_x, np.maximum(0, height-desired_height), start_x+desired_width, height))
    img = img.resize((128, 128))

    img = image.img_to_array(img)
    return img / 255.


train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    rotation_range=0.2,
    horizontal_flip=True,
    validation_split=0.2,
    #preprocessing_function=preprocess
    ) # set validation split

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42,
    color_mode='rgb',
    subset='training') # set as training data

validation_generator = train_datagen.flow_from_directory(
    train_data_dir, # same directory as training data
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    seed=42,
    color_mode='rgb',
    subset='validation') # set as validation data

STEP_SIZE_TRAIN=train_generator.n // train_generator.batch_size
STEP_SIZE_VALID=validation_generator.n // validation_generator.batch_size
print("STEP_SIZE_TRAIN: %d, STEP_SIZE_VALID: %d" %(STEP_SIZE_TRAIN, STEP_SIZE_VALID))


# Calculate class weights
counter = Counter(train_generator.classes)
max_val = float(max(counter.values()))
train_class_weights = {class_id: max_val/num_images for class_id, num_images in counter.items()}


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
    model.add(Dropout(0.7))
    
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
    #model.add(Dropout(0.7))
    
    # Output
    model.add(Flatten())
    model.add(Dense(2048, activation='relu' ))
    model.add(BatchNormalization())
    model.add(Dropout(0.7))
    model.add(Dense(9, activation='softmax'))
    # compile model
    opt = SGD( momentum=0.9, learning_rate=1e-3)
    #opt = tf.keras.optimizers.Adam(learning_rate=1e-3)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #[print(n.name) for n in tf.compat.v1.get_default_graph().as_graph_def().node]

    model.summary()
    return model

model = define_model()
# Train the last stage layers of ResNet50
model.fit(
        train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        epochs=1000,
        validation_data=validation_generator,
        validation_steps=STEP_SIZE_VALID, 
        verbose=1,
        class_weight=train_class_weights,
        shuffle=True,
        callbacks=[cp_callback, tensorboard_callback, early_stopping_callback]
        
        )

