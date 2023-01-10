# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:04:57 2020

@author: aprig
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
from sklearn.metrics import accuracy_score

xtest = np.load('.\data1_lab4\data1_xtest.npy')
ytest = np.load('.\data1_lab4\data1_ytest.npy')
xtrain = np.load('.\data1_lab4\data1_xtrain.npy')
ytrain = np.load('.\data1_lab4\data1_ytrain.npy')

#%%
#funções


def Separate_Class(_xtrain,_ytrain,_nclass,_classMaxSize):
    count = [0,0,0]
    separated = np.empty([_nclass, _classMaxSize, len(_xtrain[1,:])])
    for i in range(len(_ytrain)):
        class_value =int( _ytrain[i])
        separated[(class_value-1),count[class_value-1],:] = _xtrain[i,:]
        count[class_value-1] = count[class_value-1] + 1;
    return separated 


def Compute_var_mean(_classData):
    vm = np.empty([4,2])
    vm[0,:] = [np.var(_classData[:,0]), np.var(_classData[:,1])]
    vm[1,:] = [np.mean(_classData[:,0]), np.mean(_classData[:,1])]
    vm[2,:] = [len(_classData[:,0]), len(_classData[:,1])]
    for i in range(0,len(vm[0,:])):
        vm[3,i]= math.sqrt(vm[0,i])
    return vm

def Compute_cov(_classData):
    cov = np.cov(_classData)
    return cov

#%% 
# Dados de teste
ax = plt.figure()
s = plt.scatter(xtest[:,0],xtest[:,1],c = ytest[:])
plt.axis('equal')
legend_test =plt.legend(*s.legend_elements(),
                    loc="upper right",title="Classes")
ax.add_artist(legend_test)
plt.title("Dados de Teste")
plt.xlabel('x1')
plt.ylabel('x2')
plt.ylim(-3, 8)
plt.xlim(-8, 8)

# Dados de treino
ax = plt.figure()
s = plt.scatter(xtrain[:,0],xtrain[:,1], c = ytrain[:])
plt.axis('equal')
legend_train =plt.legend(*s.legend_elements(),
                    loc="upper right",title="Classes")
ax.add_artist(legend_train)
plt.title("Dados de Treino")
plt.xlabel('x1')
plt.ylabel('x2')
plt.ylim(-3, 8)
plt.xlim(-8, 8)

#%% separar por classes

separated = Separate_Class(xtrain,ytrain,3,50)

#%% media e variancia e covariancias por classe

vm1 = Compute_var_mean(separated[0,:,:])
vm2 = Compute_var_mean(separated[1,:,:])
vm3 = Compute_var_mean(separated[2,:,:])

cov1 = np.cov(np.transpose(separated[0,:,:]))
cov2 = np.cov(np.transpose(separated[1,:,:]))
cov3 = np.cov(np.transpose(separated[2,:,:]))

#%% Pontuação(classe/X) - Naive Bayes

# P(classe/X)= P(X/classe)*P(classe)

score1X = norm.logpdf(xtest,loc = vm1[1,:], scale = vm1[3,:]) + math.log(vm1[2,0]/len(ytrain))
score2X = norm.logpdf(xtest,loc = vm2[1,:], scale = vm2[3,:]) + math.log(vm2[2,0]/len(ytrain))
score3X = norm.logpdf(xtest,loc = vm3[1,:], scale = vm3[3,:]) + math.log(vm3[2,0]/len(ytrain))

S1X = np.empty(len( score1X[:,0]))
S2X = np.empty(len( score1X[:,0]))
S3X = np.empty(len( score1X[:,0]))

for i in range (0,len(score1X[:,0])):
    S1X[i] = score1X[i,0] + score1X[i,1];
    S2X[i] = score2X[i,0] + score2X[i,1];
    S3X[i] = score3X[i,0] + score3X[i,1];


#%% Decisão - Naive Bayes
testLabel = np.empty(len( score1X[:,0]))
for i in range (0,len(S1X)):
    testLabel[i] = np.argmax([S1X[i],S2X[i],S3X[i]])+1


#%%  Dados de teste - classes estimada - Naive Bayes

ax = plt.figure()
s = plt.scatter(xtest[:,0],xtest[:,1],c = testLabel[:])
plt.axis('equal')
legend_test =plt.legend(*s.legend_elements(),
                    loc="upper right",title="Classes")
ax.add_artist(legend_test)
plt.title("Classes Estimada - Naive Bayes")
plt.xlabel('x1')
plt.ylabel('x2')
plt.ylim(-3, 8)
plt.xlim(-8, 8)

#%% Dados Corretos(%) - Naive Bayes

ac_sc_NB = accuracy_score(ytest, testLabel)*100

#%% Pontuação (classe/X) - Bayes

score1X = multivariate_normal.logpdf(xtest,cov = cov1, mean = vm1[1,:]) + math.log(vm1[2,0]/len(ytrain))
score2X = multivariate_normal.logpdf(xtest,cov = cov2, mean = vm2[1,:]) + math.log(vm2[2,0]/len(ytrain))
score3X = multivariate_normal.logpdf(xtest,cov = cov3, mean = vm3[1,:]) + math.log(vm3[2,0]/len(ytrain))



#%% Decisão - Bayes
testLabel = np.empty(len(score1X))
for i in range (0,len(score1X)):
    testLabel[i] = np.argmax([score1X[i],score2X[i],score3X[i]])+1


#%%  Dados de teste - classes estimada - Bayes

ax = plt.figure()
s = plt.scatter(xtest[:,0],xtest[:,1],c = testLabel[:])
plt.axis('equal')
legend_test =plt.legend(*s.legend_elements(),
                    loc="upper right",title="Classes")
ax.add_artist(legend_test)
plt.title("Classes Estimada - Bayes")
plt.xlabel('x1')
plt.ylabel('x2')
plt.ylim(-3, 8)
plt.xlim(-8, 8)



#%% Dados Corretos(%) - Bayes

ac_sc_B = accuracy_score(ytest, testLabel)*100



