# Import needed packages for classification
import numpy as np
import pandas as pd

# Import packages for evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the dataset
students = pd.read_csv('data.csv', sep=';')

# # Create a new feature from u - g
# students['u_g'] = students['u'] - students['g']

# Create dataframe X with features redshift and u_g
X = students.drop('Target', axis=1)

# Create dataframe y with feature class
y = students[['Target']] 

# Sets the random seed for NumPy's random number generator to 42. 
# Setting a seed ensures that the random splitting of data is reproducible, 
# the same data points will be assigned to the training and testing sets
np.random.seed(42)

# Split data into training and test sets
# test_size=0.3: Specifies that 30% of the data should be allocated  
# to the testing set and 70% will be used for training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Initialize model with k=3

studentsKnn = KNeighborsClassifier(n_neighbors=3)
'''
without stratify: Accuracy score is 0.476
with stratify: Accuracy score is 0.428
'''

# Fit model using X_train and y_train
studentsKnn.fit(X_train, np.ravel(y_train))

# Find the predicted classes for X_test
y_pred = studentsKnn.predict(X_test)

# Calculate accuracy score
score = studentsKnn.score(X_test, np.ravel(y_test))

# Print accuracy score
print('Accuracy score is ', end="")
print('%.3f' % score)