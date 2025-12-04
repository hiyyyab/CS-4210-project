import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# load data
students = pd.read_csv('data.csv', sep=";")

# combine Graduate + Enrolled into one label
students.replace(['Graduate', 'Enrolled'], 'Not Dropout', inplace=True)

# drop features we aren't using
drop_cols = [
    'Course',
    'Admission grade',
    'Application mode',
    'Application order',
    'Curricular units 1st sem (without evaluations)',
    'Curricular units 2nd sem (without evaluations)',
    'Unemployment rate',
    'Inflation rate',
    'GDP'
]
students.drop(columns=drop_cols, inplace=True)

# split features and target
X = students.drop('Target', axis=1)
y = students['Target']

# scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# split before SMOTE so test set stays untouched
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# balance only the training data
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

# train GNB
model = GaussianNB()
model.fit(X_train_bal, y_train_bal)

# evaluate on test set
y_pred = model.predict(X_test)

# accuracy
print("Test accuracy:", round(model.score(X_test, y_test), 3))

# confusion matrix + classification report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
