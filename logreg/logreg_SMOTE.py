import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#import SMOTE
from imblearn.over_sampling import SMOTE



# read in the csv file
data = pd.read_csv("data.csv")

# classify the data into grad/dropout/enr and map it to num values
data["y"] = data["Target"].map({
    "Graduate": 0, "Dropout": 1, "Enrolled": 2
})

# see the top of the table (confirmation)
data.head()
#print(data.head())

# features and label
X = data.drop(columns=["Target", "y"])
y = data["y"]

# train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.2, random_state=42, stratify=y)
#Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)



# initialize model
logisticModel = LogisticRegression(max_iter=10000, penalty='l2')

# normalization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# fit model/ train
logisticModel.fit(X_train_scaled, y_train_resampled)

# print fitted model
#print("w1, w2:", logisticModel.coef_[0:3])
#print("w0:", logisticModel.intercept_[0:3])

# predicts the  probability of every outcome for each instance
logisticModel.predict_proba(X)[0:3]
#print(logisticModel.predict_proba(X)[0:3])

# picks class with highest probability
logisticModel.predict(X)[0:3]
#print(logisticModel.predict(X)[0:3])

# evaluate on training data
logisticModel.score(X_train_scaled, y_train_resampled)
#print("Training score is:", logisticModel.score(X_train_scaled, y_train_resampled))

# evaluate on test data
logisticModel.score(X_test_scaled, y_test)
print("Test score is:", logisticModel.score(X_test_scaled, y_test))

#confusion matrix

#test data prediction
y_pred = logisticModel.predict(X_test_scaled)

#compute matrix
cm = confusion_matrix(y_test, y_pred)

# class labels
labels = ["Graduate", "Dropout", "Enrolled"]

# plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

