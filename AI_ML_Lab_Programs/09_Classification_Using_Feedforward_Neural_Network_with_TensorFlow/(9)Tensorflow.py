# Feedforward Neural Network using TensorFlow
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Create Dataset; Features: Glucose Level, BMI, Age
X = np.array([
    [85, 22.0, 25],
    [90, 23.5, 30],
    [95, 24.0, 28],
    [100, 25.5, 35],
    [105, 26.0, 40],
    [110, 27.5, 45],
    [115, 28.0, 38],
    [120, 29.5, 50],
    [125, 30.0, 42],
    [130, 31.5, 48],
    [135, 32.0, 55],
    [140, 33.0, 52],
    [145, 34.0, 58],
    [150, 35.0, 60],
    [155, 36.0, 62],
    [160, 37.0, 65],
    [165, 38.0, 57],
    [170, 39.0, 68],
    [175, 40.0, 70],
    [180, 41.0, 72]
])

# Target: 0 = Low Risk, 1 = High Risk
y = np.array([
    0, 0, 0, 0, 0,
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1,
    1, 1, 1, 1, 1
])

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# Scale the Features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create Feedforward Neural Network
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(3,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the Model
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# Train the Model
model.fit(
    X_train_scaled,
    y_train,
    epochs=50,
    batch_size=4,
    verbose=0
)

# Evaluate the Model
loss, accuracy = model.evaluate(
    X_test_scaled,
    y_test,
    verbose=0
)

print("\n===== MODEL PERFORMANCE =====")
print("Test Accuracy:", round(accuracy * 100, 2), "%")

# Enter New Patient Data
new_patient = np.array([
    [150, 34.0, 55]
])

print("\n===== NEW PATIENT =====")
print("Glucose Level : 150")
print("BMI           : 34.0")
print("Age           : 55")

new_patient_scaled = scaler.transform(new_patient)

# Predict
prediction = model.predict(
    new_patient_scaled,
    verbose=0
)

probability = prediction[0][0]

print("\n===== PREDICTION =====")
print("Model Output:", round(float(probability), 4))

if probability >= 0.5:
    print("Diabetes Risk: High Risk")
else:
    print("Diabetes Risk: Low Risk")
