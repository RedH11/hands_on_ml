from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow.keras as keras
from tensorflow.keras.layers import Dense

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

# Build the model
model = keras.Sequential([
    # Very noisy data, so a single hidden layer is used to avoid overfitting
    Dense(30, activation="relu", input_shape=X_train.shape[1:]),

    # Only  a signle layer at the end because only a single price is being predicted as the output
    Dense(1)
])

model.compile(loss="mse", optimizer="sgd")

# Train model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid))
mse_test = model.evaluate(X_test, y_test)
X_new = X_test[:3]  # Pretending these are novel datapoints
y_pred = model.predict(X_new)

print(y_test[:3])
print(y_pred)