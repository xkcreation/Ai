# KNN using Loan Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load Dataset
data = pd.read_csv("loan_data_100_records.csv")
print("===== LOAN DATASET =====")
print(data.head())
print("\nTotal Records:", len(data))

# Step 2: Select Required Columns
data = data[
    ["ApplicantIncome", "LoanAmount",
     "Credit_History", "Loan_Status"]
]

# Step 3: Remove Missing Values
data = data.dropna()
print("\nRecords after removing missing values:", len(data))

# Step 4: Convert Target Column into Numbers
encoder = LabelEncoder()
data["Loan_Status"] = encoder.fit_transform(
    data["Loan_Status"]
)

# N = 0; Y = 1

# Step 5: Separate Features and Target
X = data[
    ["ApplicantIncome", "LoanAmount", "Credit_History"]
]

y = data["Loan_Status"]


# Step 6: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples :", len(X_test))

# Step 7: Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 8: Create KNN Model
knn = KNeighborsClassifier(n_neighbors=5)

# Step 9: Train KNN Model
knn.fit(X_train_scaled, y_train)

# Step 10: Predict Test Data
y_pred = knn.predict(X_test_scaled)
print("\n===== ACTUAL VS PREDICTED =====")
result_table = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print(result_table)

# Step 11: Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print("\n===== MODEL PERFORMANCE =====")
print("KNN Accuracy:", round(accuracy * 100, 2), "%")
print("\nConfusion Matrix:")
print(cm)

# Step 12: Test a New Applicant
new_data = pd.DataFrame(
    [[5000, 150, 1]],
    columns=[
        "ApplicantIncome",
        "LoanAmount",
        "Credit_History"
    ]
)

print("\n===== NEW APPLICANT =====")
print(new_data)

# Scale new applicant data
new_data_scaled = scaler.transform(new_data)

# Predict Loan Status
prediction = knn.predict(new_data_scaled)

# Convert prediction back to original label
loan_status = encoder.inverse_transform(prediction)

# Step 13: Display Final Prediction
if loan_status[0] == "Y":
    print("\nLoan Status: Approved")
else:
    print("\nLoan Status: Not Approved")
