# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 19:59:15 2020

@author: aprig
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm.libsvm import predict_proba

pt = pd.read_csv('data_lab4/pt_trigram_count.tsv',sep ='\t',header=None)
es = pd.read_csv('data_lab4/es_trigram_count.tsv',sep ='\t',header=None)
en = pd.read_csv('data_lab4/en_trigram_count.tsv',sep ='\t',header=None)
fr = pd.read_csv('data_lab4/fr_trigram_count.tsv',sep ='\t',header=None)

#%% shape e head


print('pt\n shape-', pt.shape, '\nhead', pt.head(1))
print('es\n shape-', es.shape, '\nhead', es.head(1))
print('en\n shape-', en.shape, '\nhead', en.head(1))
print('fr\n shape-', fr.shape, '\nhead', fr.head(1))

#%% Xtrain ytrain
X_train = np.zeros((4, pt.shape[0]))
pt_aux = pt.to_numpy()
es_aux = es.to_numpy()
en_aux = en.to_numpy()
fr_aux = fr.to_numpy()

trigrams = np.transpose(pt_aux[:,1])

X_train[0,:] = np.transpose(pt_aux[:,2])
X_train[1,:] = np.transpose(es_aux[:,2])
X_train[2,:] = np.transpose(en_aux[:,2])
X_train[3,:] = np.transpose(fr_aux[:,2])

# labels
# pt-1
# es-2
# en-3
# fr-4

y_train = np.transpose(np.array([1, 2, 3, 4]))

#%%
NB = MultinomialNB(alpha=1.0, fit_prior=False)

NB.fit(X_train, y_train)

#%%

y_confirm = NB.predict(X_train)
print('Accuracy score with training data: ', accuracy_score(y_train, y_confirm), '\n')

#%%

sentences =[
        'Que fácil es comer peras.',
        'Que fácil é comer peras.',
        'Today is a great day for sightseeing.',
        'Je vais au cinéma demain soir.',
        'Ana es inteligente y simpática.',
        'Tu vais à escola hoje.',
        'Tu vais para a escola hoje.'
        ]

vector = CountVectorizer(vocabulary = trigrams, analyzer='char', ngram_range =(3,3))

test_vector = (vector.fit_transform(sentences))
X_test = test_vector.toarray()

y_test = np.transpose(np.array([2,1,3,4,2,1,1]))

y_result = NB.predict(X_test)

print('Accuracy score with test data: ', 
          accuracy_score(y_test, y_result), '\n')

prob = np.array((NB.predict_proba(X_test)))
print('Results from predict_probaility:\n', prob)

#por por ordem crescente
prob.sort()
print('\nClassification margins')
for i in range(0, prob.shape[0]):
    print(prob[i,3]-prob[i,2])
