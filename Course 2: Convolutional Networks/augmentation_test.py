from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *
import tensorflow as tf

"""
Author: Hunter Webb
Date: 10-25-20
Summary:

Using the cats vs dogs dataset as well as augmentation to expand the diveristy of the images to improve training
even when using smaller amounts of data

"""


physical_devices = tf.config.list_physical_devices('GPU')
try:
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  # Invalid device or cannot modify virtual devices once initialized.
  pass

# Retrieving Data
train_dir = '/Users/hunterwebb/Documents/GitHub/hands_on_ml/Course 2: Convolutional Networks/cvd_train'
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.1,
    # Augmentation Settings
    rotation_range=40, # How many degrees image is randomly rotated
    width_shift_range=0.2, # Proportionate to image size, how much to move the subject horiz.
    height_shift_range=0.2, # Proportionate to image size, how much to move the subject vert.
    shear_range=0.2, # Skew the image along the x axis (in this case 20%) to make varied poses
    zoom_range=0.2, # Zooms will be random up to 20%
    horizontal_flip=True, # Images will be flipped horizontally at random
    fill_mode='nearest' # Fill mode specifies what strategy to use to fill in the pixels that are lost in transformations
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=10,
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

class AccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') >= 0.99:
            print("\nAccuracy has reached 0.99, stopping training.")
            self.model.stop_training = True

acc_cb = AccuracyCallback()

# Training Model
history = model.fit(
    train_generator,
    epochs=100,
    steps_per_epoch=100,
    validation_steps=50,
    shuffle=True,
    callbacks=[acc_cb, checkpoint_cb]
)