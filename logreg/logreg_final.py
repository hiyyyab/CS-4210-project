import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# read in the csv file
data = pd.read_csv("data.csv")

# replace all instances of 'Graduate' or 'Enrolled' with a new category, 'Not Dropout'
# we believe there is issue with model being unable to discern between graduated and
# enrolled, since they are similar ideas (both succeed at not dropping out)
data.replace(['Graduate', 'Enrolled'], 'Not Dropout', inplace=True)
# drop some features from data set deemed "non-useful"
data.drop(['Admission grade', 'Course', 'Application mode', 'Application order',
           'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (without evaluations)',
           'Unemployment rate', 'Inflation rate', 'GDP'], axis=1, inplace=True)

# see the top of the table (confirmation)
data.head()
print(data.head())

# features and label
X = data.drop(columns=["Target"])
y = data["Target"].map({"Not Dropout": 0, "Dropout": 1})

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42, stratify=y)

# initialize model
logisticModel = LogisticRegression(max_iter=10000, penalty='l2')

# normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# fit model/ train
logisticModel.fit(X_train_scaled, y_train)

# print fitted model
# print("Coefficients for every feature (w1):", logisticModel.coef_[0])
# print("Intercept (w0):", logisticModel.intercept_[0])

# predicts the  probability of every outcome for each instance
# logisticModel.predict_proba(X_test_scaled)[0:2]
# print("The predicted probability of a sample size of students not dropping out and dropping out respectively are :",logisticModel.predict_proba(X_test_scaled)[0:4])


# evaluate accuracy on test data
logisticModel.score(X_test_scaled, y_test)
print("\n Accuracy score is: ", logisticModel.score(X_test_scaled, y_test))

# evaluate  accuracy on training data
# logisticModel.score(X_train_scaled, y_train)
# print(" Training accuracy is:", logisticModel.score(X_train_scaled, y_train))


# test data prediction
y_pred = logisticModel.predict(X_test_scaled)

# f1, precision and recall scores
print("\n", classification_report(y_test, y_pred))

# confusion matrix

# compute matrix
cm = confusion_matrix(y_test, y_pred)

# class labels
labels = [" Not Dropout", "Dropout"]

# plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
