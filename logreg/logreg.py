import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# read in the csv file
data = pd.read_csv("data.csv")

# classify the data into grad/dropout/enr and map it to num values
data["y"] = data["Target"].map({
    "Graduate": 0, "Dropout": 1, "Enrolled": 2
})

# see the top of the table (confirmation)
data.head()
print(data.head())

# features and label
X = data.drop(columns=["Target", "y"])
y = data["y"]

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
print("w1, w2:", logisticModel.coef_[0:3])
print("w0:", logisticModel.intercept_[0:3])

# predicts the  probability of every outcome for each instance
logisticModel.predict_proba(X)[0:3]
print(logisticModel.predict_proba(X)[0:3])

# picks class with highest probability
logisticModel.predict(X)[0:3]
print(logisticModel.predict(X)[0:3])

# evaluate on training data
logisticModel.score(X_train_scaled, y_train)
print("Training score is:", logisticModel.score(X_train_scaled, y_train))

# evaluate on test data
logisticModel.score(X_test_scaled, y_test)
print("Test score is:", logisticModel.score(X_test_scaled, y_test))
