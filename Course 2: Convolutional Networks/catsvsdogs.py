from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import *

# Retrieving Data
train_dir = '/Users/hunterwebb/Documents/GitHub/hands_on_ml/Course 2: Convolutional Networks/cvd_train'
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.1)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
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
    metrics=['acc', 'val_accuracy']
)

# Define Callbacks
es_cb = EarlyStopping(patience=3, restore_best_weights="True")
checkpoint_cb = ModelCheckpoint('cvd_model2.h5', monitor='val_accuracy', save_best_only=True)

class AccuracyCallback(Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('acc') >= 0.99:
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
