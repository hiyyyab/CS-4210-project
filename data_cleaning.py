# NOT USED FOR RUNNING DIRECTLY
# can be copied before all code in order to load data into a dataframe
# and perform 'cleaning' (replacing of target values and dropping of features)

# Import needed packages for classification
import numpy as np
import pandas as pd

# Load the dataset
students = pd.read_csv('data.csv', sep=';')

# replace all instances of 'Graduate' or 'Enrolled' with a new category, 'Not Dropout'
    # we believe there is issue with model being unable to discern between graduated and 
    # enrolled, since they are similar ideas (both succeed at not dropping out)
students.replace(['Graduate', 'Enrolled'], 'Not Dropout', inplace=True)

# drop some features from data set deemed "non-useful"
# potentially try doing just this?
    # 'Unemployment rate', 'Inflation rate', 'GDP' all dropped - not conditional to each student
    # 'Application mode', 'Application order' - did not seem super relevant to deciding student success
    # 'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (without evaluations)'
        # above seemed nonrelevant, as units completed before university that were not 
        # taken into consideration by school seemed unlikely to have an effect
    # 'Course' - undefined meaning of column, seemed irrelevant
    # 'Admission grade' - occured before university, could be dropped
students.drop(['Course', 'Admission grade', 'Application mode', 'Application order', 'Curricular units 1st sem (without evaluations)', 'Curricular units 2nd sem (without evaluations)', 'Unemployment rate', 'Inflation rate', 'GDP'], axis=1, inplace=True)
