# -*- coding: utf-8 -*-

"""
Created on Fri Oct 30 18:15:04 2020

@author: ist

Usage: visualize_activations(CNN_model,[0,2],test_image)


"""

import matplotlib.pyplot as plt
import numpy as np
import keras


from sklearn.metrics import accuracy_score as AC

from sklearn.metrics import confusion_matrix as CM

def visualize_activations(conv_model,layer_idx,image):
    plt.figure(0)
    plt.imshow(image,cmap='gray')
    outputs = [conv_model.layers[i].output for i in layer_idx]
    
    visual = keras.Model(inputs = conv_model.inputs, outputs = outputs)
    
    features = visual.predict(np.expand_dims(np.expand_dims(image,0),3))  
        
    f = 1
    for fmap in features:
            square = int(np.round(np.sqrt(fmap.shape[3])))
            plt.figure(f)
            for ix in range(fmap.shape[3]):
                 plt.subplot(square, square, ix+1)
                 plt.imshow(fmap[0,:, :, ix], cmap='gray')
            plt.show()
            plt.pause(2)
            f +=1
            
            
def loss_plot(History,title,xlabel,ylabel):
    plt.figure()
    plt.plot(History.history['loss']) 
    plt.plot(History.history['val_loss']) 
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(['training','validation'])
    
def Pcheck_CNN(test_train, test_label, stop, title, callback):  
    if callback == 1:
        predicted = CNN_model.predict(test_train, batch_size=200, callbacks=[stop])
    else:
        predicted = CNN_model.predict(test_train, batch_size=200)
    pred = np.argmax(predicted,axis=1)
    test = np.argmax(test_label,axis=1)
    score = AC( test, pred)
    conf_matrix = CM(test, pred)
    print('\n', title)
    print('Accuracy_score=', score)
    print('Confusion_matrix = \n', conf_matrix)

def Pcheck_MLP(test_train, test_label, stop, title, callback):  
    if callback == 1:
        predicted = model_MLP.predict(test_train, batch_size=200, callbacks=[stop])
    else:
        predicted = model_MLP.predict(test_train, batch_size=200)
    pred = np.argmax(predicted,axis=1)
    test = np.argmax(test_label,axis=1)
    score = AC(test, pred)
    conf_matrix = CM(test, pred)
    print('\n', title)
    print('Accuracy_score=', score)
    print('Confusion_matrix = \n', conf_matrix)

#%% data 
data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

for i in range (1, 6):
    plt.figure()
    plt.imshow(train_images[i])
    plt.figure()
    plt.imshow(test_images[i])
    

#class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat','Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0

label = keras.utils.to_categorical(train_labels ,num_classes=10)
test_labels = keras.utils.to_categorical(test_labels ,num_classes=10)

from sklearn.model_selection import train_test_split

(train_train_images, test_train_images, train_train_labels, test_train_lables) = train_test_split(train_images, label, test_size=0.20, random_state=42)


train_train_images = np.expand_dims(train_train_images, 3)
test_train_images = np.expand_dims(test_train_images, 3)
test_images = np.expand_dims(test_images, 3)



#%% MLP

model_MLP = keras.Sequential()

#First Layer
model_MLP.add(keras.layers.Flatten(input_shape=(28,28,1), name='layer_one'))

#Second Layer
model_MLP.add(keras.layers.Dense(32, activation='relu', name='hidden_one'))

#Third Layer
model_MLP.add(keras.layers.Dense(64, activation='relu', name='hidden_two'))

#Fourt Layer
model_MLP.add(keras.layers.Dense(10, activation='softmax', name='layer_four'))

#summary
model_MLP.summary()

#Early Stop
MLP_stop=keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

#Compile
model_MLP.compile('Adam', loss='categorical_crossentropy')

#%%With Early stopping

#Fitting

history_obj=model_MLP.fit(x=train_train_images, y=train_train_labels, batch_size=200, epochs=200, verbose=0,
              callbacks=[MLP_stop],validation_data=(test_train_images, test_train_lables))

#Plot
loss_plot(history_obj,'MLP loss with early stop','Epoch','Loss');

#Evaluate Perfomance
Pcheck_MLP(test_images, test_labels, MLP_stop,'MLP with Early Stopping',0);

#%% MLP

model_MLP = keras.Sequential()

#First Layer
model_MLP.add(keras.layers.Flatten(input_shape=(28,28,1), name='layer_one'))

#Second Layer
model_MLP.add(keras.layers.Dense(32, activation='relu', name='hidden_one'))

#Third Layer
model_MLP.add(keras.layers.Dense(64, activation='relu', name='hidden_two'))

#Fourt Layer
model_MLP.add(keras.layers.Dense(10, activation='softmax', name='layer_four'))

#summary
model_MLP.summary()

#Early Stop
MLP_stop=keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

#Compile
model_MLP.compile('Adam', loss='categorical_crossentropy')

#%%Without Early stopping

#Fitting

history_obj=model_MLP.fit(x=train_train_images, y=train_train_labels, batch_size=200, epochs=200, verbose=0,
              callbacks=None,validation_data=(test_train_images, test_train_lables))

#Plot
loss_plot(history_obj,'MLP loss without early stop','Epoch','Loss');

#Evaluate Perfomance
Pcheck_MLP(test_images, test_labels, MLP_stop,'MLP without Early Stopping',0);
#%%CNN

# Modelo

CNN_model = keras.Sequential()

# First Layer
CNN_model.add(keras.layers.Conv2D(16, kernel_size = (3, 3), activation= 'relu', input_shape = (28, 28, 1)))

# Second Layer
CNN_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Third Layer
CNN_model.add(keras.layers.Conv2D(16, kernel_size=(3, 3), activation= 'relu'))

# Fourth Layer
CNN_model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

# Fifth Layer
CNN_model.add(keras.layers.Flatten())

# Sixth Layer
CNN_model.add(keras.layers.Dense(32, activation = 'relu'))

# Seventh Layer
CNN_model.add(keras.layers.Dense(10, activation='softmax'))

# Sumário do Modelo

CNN_model.summary()


# Early stop
CNN_stop = keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)

# Compile
CNN_model.compile('Adam', loss='categorical_crossentropy')

# Fit
History_CNN = CNN_model.fit(x=train_train_images, y=train_train_labels, batch_size=200, epochs=200, verbose=0, callbacks=[CNN_stop], validation_data=(test_train_images,test_train_lables))

#%% CNN plot

loss_plot(History_CNN,'CNN loss with early stop','Epoch','Loss');

Pcheck_CNN(test_images, test_labels, CNN_stop,'CNN with Early Stopping',0);

#%% Visualize the feature maps obtained

idx =np.array([0, 2])

for i  in range(1, 3):
    visualize_activations(CNN_model,idx,test_images[i, :, :, 0])




