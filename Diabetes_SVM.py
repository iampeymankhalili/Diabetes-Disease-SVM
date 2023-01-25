import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score , recall_score , balanced_accuracy_score
from sklearn.metrics import roc_curve , auc
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression



# Determining the path of the database file to read information from it
absolute_path = os.path.abspath(os.path.dirname('diabetes.csv'))

Diabetes_data = pd.read_csv(os.path.join(os.path.dirname(__file__), "../Diabetes-Disease-SVM/diabetes.csv"))


x = Diabetes_data.iloc[:,0:-1].values
y = Diabetes_data.iloc[:,-1].values

# Train-Test split
# use train_test_split from sklearn.model_selection
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.20, random_state = 4)

# Feature Scaling
# use StandardScaler from sklearn.preprocessing
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)


# SVM
# use SVC from sklearn.svm
model_SVC = SVC(kernel = 'rbf', random_state = 4)
model_SVC.fit(x_train, y_train)

y_prediction_svm = model_SVC.decision_function(x_test)

# logistic regression
# use LogisticRegression from sklearn.linear_model
model_lr = LogisticRegression()
model_lr.fit(x_train, y_train)

y_prediction_logistic = model_lr.decision_function(x_test)


# Draw the ROC and AUC Plot as mentioned in the article.
# use roc_curve , auc from sklearn.metrics
# and use plt fom matplotlib.pyplot

logistic_fpr, logistic_tpr, threshold = roc_curve(y_test, y_prediction_logistic)
auc_logistic = auc(logistic_fpr, logistic_tpr)

svm_fpr, svm_tpr, threshold = roc_curve(y_test, y_prediction_svm)
auc_svm = auc(svm_fpr, svm_tpr)

plt.figure(figsize=(5, 5), dpi=100)
plt.plot(svm_fpr, svm_tpr, linestyle='-', label='SVM (auc = %0.3f)' % auc_svm)
plt.plot(logistic_fpr, logistic_tpr, marker='.', label='Logistic (auc = %0.3f)' % auc_logistic)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.legend()

plt.show()

# accuracy_score of training data

x_train_prediction_accuracy=model_SVC.predict(x_train)
training_data_accuracy_score=accuracy_score(y_train,x_train_prediction_accuracy)
print(f"Accuracy Score of training data : {training_data_accuracy_score * 100} %")

# accuracy_score of testing data

x_test_prediction_accuracy=model_SVC.predict(x_test)
testing_data_accuracy_score=accuracy_score(y_test,x_test_prediction_accuracy)
print(f"Accuracy Score of testing data : {testing_data_accuracy_score * 100} %")

# recall_score of training data

x_train_prediction_recall=model_SVC.predict(x_train)
training_data_recall_score=recall_score(y_train,x_train_prediction_recall)
print(f"recall(sensitivity) Score of training data : {training_data_recall_score * 100} %")

# recall_score of testing data

x_test_prediction_recall=model_SVC.predict(x_test)
testing_data_recall_score=recall_score(y_test,x_test_prediction_recall)
print(f"recall(sensitivity) Score of testing data : {testing_data_recall_score * 100} %")

# balanced_score of training data

x_train_prediction_balanced=model_SVC.predict(x_train)
training_data_balanced_score=balanced_accuracy_score(y_train,x_train_prediction_balanced)

# balanced_score of testing data

x_test_prediction_balanced=model_SVC.predict(x_test)
testing_data_balanced_score=balanced_accuracy_score(y_test,x_test_prediction_balanced)

# specificity = (2*balanced) - (recall)
# specificity_score of training data

training_data_specificity_score=(2 * training_data_balanced_score) - (training_data_recall_score)
print(f"specificity Score of training data : {training_data_specificity_score * 100} %")

# specificity_score of testing data

testing_data_specificity_score=(2 * testing_data_balanced_score) - (testing_data_recall_score)
print(f"specificity Score of testing data : {testing_data_specificity_score * 100} %")