"""
Aprendizagem Automática - Trabalho 5 - Tarefa de Classificação

@author: Alice Rosa 90007 
         Aprígio Malveiro 90026
         G1 - Terça 14h00
"""
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix as cm
from sklearn.metrics import precision_score
from sklearn.naive_bayes import GaussianNB

def metrics(true_y,pred_y):
    
    acc_score=accuracy_score(true_y,pred_y)
    print("\n Accuracy Score:", acc_score)
    
    balanced_acc_score=balanced_accuracy_score(true_y,pred_y)
    print("\n Balanced Accuracy Score:", balanced_acc_score)
    
    f_measure=f1_score(true_y,pred_y)
    print("\n F-Measure:", f_measure)
    
    prec_score = precision_score(true_y, pred_y)
    print('\n Precision: ', prec_score)
    
    conf_mat = cm(true_y, pred_y)
    print('\n Confusion Matrix: \n', conf_mat)
   
    return 

#%% Load and pre-process data

X_train = np.load('Cancer_Xtrain.npy')
y_train = np.load('Cancer_ytrain.npy')
X_test = np.load('Cancer_Xtest.npy')
y_test = np.load('Cancer_ytest.npy')

min_max_scaler = preprocessing.MinMaxScaler()
x_train = min_max_scaler.fit_transform(X_train)
x_test = min_max_scaler.transform(X_test)


#%% SVM 

tuned_parameters = [{'kernel': ['rbf'], 'gamma': np.logspace(-5, 3, 20),
                     'C': [1, 10, 50, 100, 500, 1000, 10000]},
                    {'kernel': ['linear'], 'C': [1, 10, 50, 100, 500, 1000, 10000]},
                    {'kernel': ['poly'],'degree':np.arange(1,9), 'C': [1, 10, 50, 100, 500, 1000, 10000]}]

clf=GridSearchCV(SVC(), tuned_parameters, cv=6)
clf.fit(x_train, np.ravel(y_train))

print("\n Best parameters: ", clf.best_params_)

# Testing
y_true, test_y_pred = y_test, clf.predict(x_test)
print('\n Metrics for the test set:')
metrics(y_true, test_y_pred)


#%% Naive Bayes

gnb = GaussianNB()

y_pred = gnb.fit(x_train, np.ravel(y_train)).predict(x_test)

metrics(y_test,y_pred)
