# Import needed packages for classification
import numpy as np
import pandas as pd

# Import packages for evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# Load the dataset
students = pd.read_csv('data.csv', sep=';')

# replace all instances of 'Graduate' or 'Enrolled' with a new category, 'Not Dropout'
    # we believe there is issue with model being unable to discern between graduated and 
    # enrolled, since they are similar ideas (both succeed at not dropping out)
students.replace(['Graduate', 'Enrolled'], 'Not Dropout', inplace=True)

# drop some features from data set deemed "non-useful"
students.drop(['Admission grade', 'Course', 'Application mode', 'Application order', 'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (without evaluations)', 'Unemployment rate', 'Inflation rate', 'GDP'], axis=1, inplace=True)

print(students.head())

# Create dataframe X with all features besides target
X = students.drop(['Target'], axis=1)

# Create dataframe y with feature class
y = students[['Target']] 

# Split data into training and test sets
# test_size=0.2: researchers specify that 20% of the data should be allocated  
# to the testing set and 80% will be used for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Elbow Method used to find potential values for k
k_values = range(1, 67)
error_rates = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, np.ravel(y_train))
    y_pred = knn.predict(X_test)
    error_rate = 1 - accuracy_score(y_test, y_pred)
    error_rates.append(error_rate)

plt.figure(figsize=(8, 6))
plt.plot(k_values, error_rates, marker='o')
plt.xlabel('K Value')
plt.ylabel('Error Rate')
plt.title('Elbow Method for Finding Optimal K')
plt.axvline(x=7, linestyle='--', color='red', label="Optimal K (Elbow Point)")
plt.legend()
plt.show()

# Initialize model with k = 7 (tested a couple others: 3, 5, 7, 9)
studentsKnn = KNeighborsClassifier(n_neighbors=7, weights='distance')

# Fit model using X_train and y_train
studentsKnn.fit(X_train, np.ravel(y_train))

# Find the predicted classes for X_test
y_pred = studentsKnn.predict(X_test)

# Calculate accuracy score
# note that it can be misleading with imbalanced data, should check others (done below)
score = studentsKnn.score(X_test, np.ravel(y_test))

# Print accuracy score
print('Accuracy score is ', end="")
print('%.3f' % score)
    # k = 3 (first drop) gave accuracy = 0.817
    # k = 5 (second drop) gave accuracy = 0.821
    # k = 7 (third drop) gave accuracy = 0.824
    # k = 9 (check outer bounds) gave accuracy = 0.821

# gives info about precision, recall, and F1-score
print(classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Dropout', 'Not Dropout'])
disp.plot()
plt.show()

