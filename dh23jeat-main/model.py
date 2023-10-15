import tensorflow as tf
from tensorflow import keras

# Assuming X_train, y_train, X_val, y_val, X_test, and new_data are properly defined

# Define your model
model = keras.Sequential()

# Input layer
model.add(keras.layers.Input(shape=(48,)))

# Hidden layers
model.add(keras.layers.Dense(units=64, activation='relu'))
model.add(keras.layers.Dense(units=32, activation='relu'))

# Output layer
model.add(keras.layers.Dense(units=1, activation='linear'))

# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])

# Train the model
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model
loss, mae = model.evaluate(X_test, y_test)
print(f"Mean Absolute Error: {mae}")

# Make predictions
predictions = model.predict(new_data)
