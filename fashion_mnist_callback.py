"""
Hunter Webb
8/31/20

Training a simple neural network on the fashion_mnist dataset while implementing the callback
feature as taught in the coursera week 2 introduction to tensorflow course
"""

import tensorflow.keras as keras
from tensorflow.keras.layers import Flatten, Dense
import pandas as pd
import matplotlib.pyplot as plt
from FashionCallback import FashionCallback

callback = FashionCallback()
# Retrieving the labelled dataset
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# Scaling features and creating validation set
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test / 255.0

# Defining the meaning of the y values
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# Creating the model
model = keras.models.Sequential([
    # Could alternatively be defined as InputLayer(), but both do the same preprocessing to
    # make the 2d black and white image vector into a 1d vector
    Flatten(input_shape=[28, 28]),

    # The dense layers are fully connected layers with each layer managing a weights matrix and a vector of biases
    Dense(300, activation="relu"),
    Dense(100, activation="relu"),

    # The softmax activation is used because the classes are exclusive and
    # it return a probability array for the model's confidence
    Dense(10, activation="softmax")
])

model.compile(
    # Sparse categorical crossentropy is used because the result is a single number representing the correct
    # classification (ex. [1]) instead of categorical crossentropy which would have a desired probability array
    # for the result (ex. [0.0, 0.0, 1.0, 0.0])
    loss="sparse_categorical_crossentropy",

    # Keras will use stochastic gradient descent for back-propogation
    # and when using sgd it is usually important to tune the learning rate which
    # is by default set to 0.01
    optimizer="sgd",

    # For classification, it is useful to measure the accuracy for evaluation
    metrics=['accuracy']
)

# Training the model
history = model.fit(X_train, y_train, epochs=30,
                    # Instead of passing in a validation set, you can use validation_split=(0.0-1.0) to
                    # specify what % to use as the validation set
                    validation_data=(X_valid, y_valid),
                    # Using callback to halt training when the loss reaches less than 0.4
                    callbacks=[callback]
)

# The model is trained! Let's evaluate it
model.evaluate(X_test, y_test)

# Graphing the model's history of accuracy
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)  # set the vertical range to [0-1]
plt.show()


