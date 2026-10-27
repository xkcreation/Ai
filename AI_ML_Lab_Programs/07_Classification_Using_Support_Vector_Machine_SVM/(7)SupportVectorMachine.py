from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

# Load Iris Dataset
iris = load_iris()

X = iris.data
y = iris.target

print("===== IRIS DATASET =====")
print("Classes:", iris.target_names)
print("Total Samples:", len(X))

# Randomly split the dataset
# 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

print("\n===== TRAINING AND TESTING =====")
print("Training Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# Create SVM Model
model = SVC(kernel='linear')

# Train the Model
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

print("\n===== PREDICTION =====")
print("Actual Classes   :", y_test)
print("Predicted Classes:", y_pred)

# Performance Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
cm = confusion_matrix(y_test, y_pred)

# Display Results
print("\n===== PERFORMANCE =====")
print("Accuracy  :", round(accuracy * 100, 2), "%")
print("Precision :", round(precision, 4))
print("Recall    :", round(recall, 4))
print("F1-Score  :", round(f1, 4))

print("\n===== CONFUSION MATRIX =====")
print(cm)
