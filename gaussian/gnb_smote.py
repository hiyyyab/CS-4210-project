import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# load dataset
students = pd.read_csv('data.csv', sep=";")

# create X and y
X = students.drop('Target', axis=1)
y = students['Target']

print("Original class distribution:")
print(y.value_counts(), "\n")

# train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print("Train distribution BEFORE SMOTE:")
print(y_train.value_counts(), "\n")

# apply SMOTE only on training data
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("Train distribution AFTER SMOTE:")
print(pd.Series(y_train_bal).value_counts(), "\n")

# initialize GNB
gnb = GaussianNB()

# fit on balanced training data
gnb.fit(X_train_bal, y_train_bal)

# predict on test set (original distribution)
y_pred = gnb.predict(X_test)

# print accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc:.3f}")

# print confusion matrix & classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
