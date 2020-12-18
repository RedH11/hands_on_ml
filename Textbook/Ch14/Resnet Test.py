import tensorflow.keras as keras
import tensorflow as tf
import os, time, shutil
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Flatten, Dense
from Resnet34 import ResidualUnit
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.callbacks import *
"""
Author: Hunter Webb
Date: 12-17-20

Implementing Restnet34 on Bee vs Wasp Dataset

"""

input_w = 224
input_h = 224

physical_devices = tf.config.list_physical_devices('CPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

run_id = time.strftime("beesVwasp_%m_%d_%H_%M")

"""
    Retrieving Data / Making Image Generators
"""

train_dir = os.getcwd() + '/train'
training_datagen = ImageDataGenerator(rescale=1./255)
training_generator = training_datagen.flow_from_directory(
    train_dir,
    target_size=(input_w, input_h),
    batch_size=3,
    class_mode='binary'
)

testing_dir = os.getcwd() + '/test'
testing_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = testing_datagen.flow_from_directory(
    testing_dir,
    target_size=(input_w, input_h),
    batch_size=3,
    class_mode='binary'
)

"""
    Creating the model
"""

model = keras.models.Sequential()
model.add(Conv2D(64, 7, strides=2, input_shape=[input_w, input_h, 3],
                 padding="same", use_bias=False))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPool2D(pool_size=3, strides=2, padding="same"))
prev_filters = 64

# Makes three layers with 64 filters, four with 128, six with 256, and three with 512
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters

model.add(GlobalAveragePooling2D())
model.add(Flatten())
# Softmax activation for classification
model.add(Dense(10, activation='softmax'))

"""
    Creating callbacks / compiling model
"""
# Define Callbacks
checkpoint_cb = ModelCheckpoint('beesVwasps.h5', monitor='accuracy', save_best_only=True)
tensorboard_cb = TensorBoard(os.getcwd() + '/logs')

class AccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.85:
            print("\nAccuracy has reached 0.85, stopping training.")
            self.model.stop_training = True

acc_cb = AccuracyCallback()

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer=keras.optimizers.SGD(lr=0.2, momentum=0.9, decay=0.001),
    metrics=["accuracy"]
)

"""
    Training the Model
"""

history = model.fit(
    training_generator,
    validation_data = validation_generator,
    epochs=100,
    steps_per_epoch=100,
    shuffle=True,
    callbacks=[acc_cb, checkpoint_cb]
)