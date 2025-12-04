# Import the necessary modules
import numpy as np
import pandas as pd

from sklearn.naive_bayes import GaussianNB

# Load the dataset
students = pd.read_csv('data.csv')

# Create dataframe X 
X = students.drop('Target', axis=1)

# Create dataframe y 
y = students[['Target']] 

# Initialize a Gaussian naive Bayes model
skySurveyNBModel = GaussianNB()

# Fit the model
skySurveyNBModel.fit(X, np.ravel(y))

# Calculate the proportion of instances correctly classified
# skySurveyNBModel.predict(X)
score = skySurveyNBModel.score(X, np.ravel(y)) 

# Print accuracy score
print('Accuracy score is ', end="")
print('%.3f' % score)
# Accuracy score is 0.687