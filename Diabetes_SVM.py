import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve , auc
from sklearn.svm import SVC



# Determining the path of the database file to read information from it
absolute_path = os.path.abspath(os.path.dirname('diabetes.csv'))

Diabetes_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "../Diabetes-Disease-SVM/diabetes.csv"))


x = Diabetes_data.iloc[:,0:-1].values
y = Diabetes_data.iloc[:,-1].values

# Train-Test split
# use train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 4)

# Feature Scaling
# use StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)
