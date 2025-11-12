# Import needed packages for classification
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import packages for evaluation
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Load the dataset
students = pd.read_csv('data.csv', sep=';')

'''
# ** Section to find optimal K - Elbow Method ** 
# ** Gives k = 2.7 or k = 10.9 **

from sklearn.metrics import accuracy_score

X = students.drop('Target', axis=1)
y = students[['Target']] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

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
plt.axvline(x=3, linestyle='--', color='red', label="Optimal K (Elbow Point)")
plt.legend()
plt.show()
'''

'''
# ** Section to find optimal K - Grid Search with Cross-Validation ** 
# ** Gives k = 9 **

from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

X = students.drop('Target', axis=1)
y = students[['Target']] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn = KNeighborsClassifier()
param_grid = {'n_neighbors': range(1, 67)}

grid_search = GridSearchCV(knn, param_grid, cv=5)
grid_search.fit(X_train, np.ravel(y_train))

optimal_k = grid_search.best_params_['n_neighbors']
print(f"The optimal K value is: {optimal_k}")
'''

# Create dataframe X 
X = students.drop('Target', axis=1)

# Create dataframe y 
y = students[['Target']] 

# Trying to view the graph of plots... 
p = sns.scatterplot(data=students, x='Unemployment rate', y='GDP',
                    hue='Target', style='Target')
p.set_xlabel('Unemployment rate', fontsize=14)
p.set_ylabel('GDP', fontsize=14)
p.legend(title='Target')
plt.show()

# Split data into training and test sets
# random=42: Sets the random seed for NumPy's random number generator to 42, ensuring the random split of data is reproducible
# test_size=0.3: Specifies that 30% of the data for testing set and 70% for training set 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.3)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

# Initialize model with k=3
# studentsKnn = KNeighborsClassifier(n_neighbors=9)
studentsKnn = KNeighborsClassifier(n_neighbors=9, weights='distance')
'''
k = 3
without stratify: Accuracy score is 0.587
with stratify: Accuracy score is 0.584

k = 9 
without stratify: Accuracy score is 0.605
with stratify: Accuracy score is 0.614

k = 9 and weights='distance'
without stratify: Accuracy score is 0.608
with stratify: Accuracy score is 0.622

k = 11 
without stratify: Accuracy score is 0.610
with stratify: Accuracy score is 0.614

according to Google k = sqrt(4424)
trying with k = 66
without stratify: Accuracy score is 0.565
with stratify: Accuracy score is 0.578
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
print()
print(students['Target'].value_counts())

# # Gives info about precision, recall, and F1-score
# print(classification_report(y_test, y_pred))

# # Confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Dropout', 'Enrolled', 'Graduate'])
# disp.plot()
# plt.show()