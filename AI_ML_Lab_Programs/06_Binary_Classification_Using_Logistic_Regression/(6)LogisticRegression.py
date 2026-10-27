from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Training Data
# -----------------------------
train_emails = [
    "Congratulations! You won a free lottery",
    "Claim your prize now",
    "Limited time offer",
    "Meeting at 10 AM tomorrow",
    "Project report submission",
    "Please attend the class today"
]

# 1 = Spam, 0 = Not Spam
train_labels = [1, 1, 1, 0, 0, 0]

print("===== TRAINING DATA =====")
for email, label in zip(train_emails, train_labels):
    print(email, "-->", "Spam" if label == 1 else "Not Spam")

# Convert text into numerical features
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(train_emails)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, train_labels)

# -----------------------------
# Testing Data
# -----------------------------
test_email = ["Free prize available now"]

print("\n===== TESTING DATA =====")
print(test_email[0])

X_test = vectorizer.transform(test_email)

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(X_test)

print("\n===== PREDICTION =====")
print(prediction)

# -----------------------------
# Result
# -----------------------------
print("\n===== RESULT =====")
if prediction[0] == 1:
    print("The email is classified as SPAM.")
else:
    print("The email is classified as NOT SPAM.")
