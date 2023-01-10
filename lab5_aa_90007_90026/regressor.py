"""
Aprendizagem Automática - Trabalho 5 - Tarefa de Regressão

@author: Alice Rosa 90007 
         Aprígio Malveiro 90026
         G1 - Terça 14h00
"""


import matplotlib.pyplot as plt
import numpy as np
import keras

Xtest = np.load('Real_Estate_Xtest.npy')
Xtrain = np.load('Real_Estate_Xtrain.npy')
ytest = np.load('Real_Estate_ytest.npy')
ytrain = np.load('Real_Estate_ytrain.npy')


from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from sklearn import preprocessing

from sklearn.metrics import mean_squared_error 
from sklearn.metrics import max_error
from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error 
from sklearn.metrics import r2_score 

def loss_plot(History,title,xlabel,ylabel):
    plt.figure()
    plt.plot(History.history['loss']) 
    plt.plot(History.history['val_loss']) 
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(['training','validation'])
    
def acuracyScoresReg(y_true, y_predict):
    print('\n Metrics results')
    maxE = max_error(y_true, y_predict)
    print('\n Max Error:', maxE)
    
    mse = mean_squared_error(y_true, y_predict)
    print('\n Mean Squared Error:', mse)
    mae = mean_absolute_error(y_true, y_predict)
    print('\n Mean Absolute Error:', mae)
    
    evScore = explained_variance_score(y_true, y_predict)
    print('\n Explained Variance Regression Score:', evScore)
    r2Score = r2_score(y_true, y_predict)
    print('\n Regression score function:',r2Score)
    

#%% MLP Regression
    
n=Xtrain.shape[1];

min_max_scaler = preprocessing.MinMaxScaler()

xtrain = min_max_scaler.fit_transform(Xtrain)
xtest = min_max_scaler.transform(Xtest)
# Create the model
model = Sequential()

#init layer
model.add(Dense(n, input_dim=n, kernel_initializer='normal', activation='relu'))

#hidden layers
model.add(Dense(n, activation='relu'))

# linear layer output
model.add(Dense(1, activation='linear'))

#summary
model.summary()


# Configure the model and start training

MLP_stop=keras.callbacks.EarlyStopping(monitor = 'val_loss',patience=10, restore_best_weights=True)
model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_squared_error'])

loss = model.fit(xtrain, ytrain, epochs=200, batch_size=1, verbose=0,callbacks=[MLP_stop], validation_split=0.2)


loss_plot(loss,'MLP loss with early stop','Epoch','Loss');


# Acuracy Score

ypredict = model.predict(xtest, batch_size=10, callbacks=None)

print('\nScore test MLP Regressor:')
acuracyScoresReg(ytest, ypredict)

#%% Linear Regression - Least Square


ones = np.ones((Xtrain.shape[0],1))
LXtrain = np.concatenate((ones, Xtrain), axis=1)

ones = np.ones((Xtest.shape[0],1))
LXtest = np.concatenate((ones, Xtest), axis=1)


X=LXtrain

XT = np.transpose(X)
XTX = np.matmul(XT,X)
XTy = np.matmul(XT,ytrain)
B = np.linalg.solve(XTX,XTy)

ypredict = np.matmul(LXtest,B)

print('\nScore test Linear Regressor:')
acuracyScoresReg(ytest, ypredict)

