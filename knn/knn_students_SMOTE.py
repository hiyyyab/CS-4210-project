# NOTE: SMOTE was deemed inefficient for knn here, stratify=y did job well enough
# found SMOTE to potentially be creating instances that were not as accurate to the data
# and therefore was not used for remainder of KNN evaluation

# Import needed packages for classification
import numpy as np
import pandas as pd

# Import packages for evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# for Synthetic Minority Over-sampling Technique (helps model deal with imbalanced data set)
from imblearn.over_sampling import SMOTE

# Load the dataset
students = pd.read_csv('data.csv', sep=';')

# Create dataframe X with all features besides target
X = students.drop(['Target'], axis=1)

# Create dataframe y with feature class
y = students[['Target']] 

# Sets the random seed for NumPy's random number generator to 42. 
# Setting a seed ensures that the random splitting of data is reproducible, 
# the same data points will be assigned to the training and testing sets
smote = SMOTE(random_state=42)

# Split data into training and test sets
# test_size=0.3: Specifies that 30% of the data should be allocated  
# to the testing set and 70% will be used for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# only apply resampling to training data (doing so for testing will skew results)
X_resampling, y_resampling = smote.fit_resample(X_train, y_train)

# Initialize model with k=9 (tested a couple others, 9 had 'best')
studentsKnn = KNeighborsClassifier(n_neighbors=9, weights='distance')

# Fit model using X_train and y_train
studentsKnn.fit(X_resampling, np.ravel(y_resampling))

# Find the predicted classes for X_test
y_pred = studentsKnn.predict(X_test)

# Calculate accuracy score
# note that it can be misleading with imbalanced data, should check others (done below)
score = studentsKnn.score(X_test, np.ravel(y_test))

# Print accuracy score
print('Accuracy score is ', end="")
print('%.3f' % score)

# gives info about precision, recall, and F1-score
print(classification_report(y_test, y_pred))

# confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Dropout', 'Enrolled', 'Graduate'])
disp.plot()
plt.show()