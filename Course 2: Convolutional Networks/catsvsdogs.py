from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
import tensorflow as tf
import os
import time
from PIL.Image import core as _imaging

"""
    Setup
        - Allowing dynamic memory growth
        - Finding the root path to the dir
"""
physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

root_logdir = os.path.join(os.curdir, "../my_logs")

def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()

"""
    Configuring the image generators 
"""
train_dir = os.getcwd() + '/cvd_train'
training_datagen = ImageDataGenerator(rescale=1./255)
training_generator = training_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=3,
    class_mode='binary'
)

testing_dir = os.getcwd() + '/cvd_test'
testing_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = testing_datagen.flow_from_directory(
    testing_dir,
    target_size=(150, 150),
    batch_size=3,
    class_mode='binary'
)

# Building Model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),

    # Output layer, sigmoid for binary classification
    Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=['accuracy']
)

# Define Callbacks
es_cb = EarlyStopping(patience=3, monitor='loss', restore_best_weights="True")
checkpoint_cb = ModelCheckpoint('cvd_model.h5', monitor='accuracy', save_best_only=True)
tensorboard_cb = TensorBoard(run_logdir)

class AccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.99:
            print("\nAccuracy has reached 0.99, stopping training.")
            self.model.stop_training = True

acc_cb = AccuracyCallback()

# Training Model
history = model.fit(
    training_generator,
    validation_data = validation_generator,
    epochs=100,
    steps_per_epoch=100,
    shuffle=True,
    callbacks=[acc_cb, checkpoint_cb, tensorboard_cb]
)

"""
Bug Note: (For save best only) with validation split in the training image generator
Setting it to monitor accuracy somehow worked instead of val_accuracy and the accuracy in the metrics also
had to be 'accuracy' instead of 'acc'

It also seems like adding the es_cb breaks it for some reason
"""