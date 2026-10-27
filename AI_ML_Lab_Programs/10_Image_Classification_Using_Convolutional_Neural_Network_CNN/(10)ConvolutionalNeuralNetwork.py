# Convolutional Neural Network using TensorFlow
# Handwritten Digit Classification using MNIST Dataset

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random

# Load MNIST Dataset
(X_train, y_train), (X_test, y_test) = \
    tf.keras.datasets.mnist.load_data()

# Display Dataset Summary
print("===== MNIST DATASET SUMMARY =====")
print("Total Training Images :", len(X_train))
print("Total Testing Images  :", len(X_test))
print("Image Size            :", X_train.shape[1], "x", X_train.shape[2])
print("Number of Classes     : 10")
print("Classes               : 0 to 9")

# Display Sample Images
plt.figure(figsize=(8, 4))

for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X_train[i], cmap="gray")
    plt.title("Digit: " + str(y_train[i]))
    plt.axis("off")

plt.suptitle("Sample MNIST Handwritten Digits")
plt.tight_layout()
plt.show()

# Use Smaller Dataset for Faster Execution
X_train = X_train[:2000]
y_train = y_train[:2000]

X_test = X_test[:500]
y_test = y_test[:500]

print("\n===== DATA USED =====")
print("Training Images :", len(X_train))
print("Testing Images  :", len(X_test))

# Normalize Images
X_train = X_train / 255.0
X_test = X_test / 255.0

# Add Channel Dimension
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Create CNN Model
model = tf.keras.Sequential([

    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        16,
        (3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        (2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])

# Compile Model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train Model
print("\n===== MODEL TRAINING =====")

model.fit(
    X_train,
    y_train,
    epochs=2,
    batch_size=64
)

# Evaluate Model
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=0
)

print("\n===== MODEL PERFORMANCE =====")
print(
    "Test Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

# Select a Random Test Image
index = random.randint(
    0,
    len(X_test) - 1
)

# Predict the Random Image
prediction = model.predict(
    X_test[index:index + 1],
    verbose=0
)

predicted_digit = np.argmax(
    prediction[0]
)

actual_digit = y_test[index]

# Display Prediction
print("\n===== DIGIT PREDICTION =====")
print("Test Image No. :", index)
print("Actual Digit   :", actual_digit)
print("Predicted Digit:", predicted_digit)

# Display the Random Test Image
plt.imshow(
    X_test[index].reshape(28, 28),
    cmap="gray"
)

plt.title(
    "Actual: " + str(actual_digit) +
    "\nPredicted: " + str(predicted_digit)
)

plt.axis("off")
plt.show()
