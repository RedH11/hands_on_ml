import tensorflow.keras as keras
from tensorflow.keras.layers import Concatenate, Dense, Input
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Fetch the dataset
housing = fetch_california_housing()

# Assign the input and target result values
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target
)
X_train,X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full
)

# Scale the inputs
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test =  scaler.transform(X_test)
# Building the complex model (Wide and Deep) with Keras's Functional API

# Specify the shape of our input
input_ = Input(shape=X_train.shape[1:])
# Build the hidden layers using the Functional API formatting to show at the end how to connect the layers
hidden1 = Dense(30, activation="relu")(input_)
hidden2 = Dense(30, activation="relu")(hidden1)
# Combining both values together (ex. 123 and 456 concatenate to 123456)
concat = Concatenate()([input_, hidden2])
output = Dense(1)(concat)
model = keras.Model(inputs=[input_], outputs=[output])

model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))

model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]  # Pretending these are novel datapoints
y_pred = model.predict(X_new)
print("Actual:", y_test[:3], "\n\nPredicted:", y_pred)
