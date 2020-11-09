#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import tensorflow as tf
from lenet import *
from keras.optimizers import SGD
#from keras import optimizers
#from keras.optimizers import adam
from keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np



#importing the data
print("Accessing MNIST")
((trainData ,trainLabels) ,(testData, testLabels)) =mnist.load_data()

#after downloading the data we have to reshape it 
if K.image_data_format() =="channels_first":
    trainData = trainData.reshape((trainData.shape[0], 1,28,28))
    testData =testData.reshape((testData.shape[0] , 1,28,28))
    #otherwise we will use channels last for this
    
else:
    trainData = trainData.reshape((trainData.shape[0] , 28,28,1))
    testData = testData.reshape((testData.shape[0], 28,28,1))
    
    #scale our data from [0,1]
    
trainData =trainData.astype("float32")/255.0
testData =testData.astype("float32")/255.0

#convert the labels from integers to vectors

le  =LabelBinarizer()
trainLabels =le.fit_transform(trainLabels)
testLabels =le.fit_transform(testLabels)

print("INFO compiling model...")
opt = tf.keras.optimizers.SGD(lr=1e-3, momentum=0.3, decay=0, nesterov=False)
model = LeNet.build(width=28, height=28, depth=1, classes=10)
model.compile(loss="categorical_crossentropy" , optimizer=opt,
metrics=["accuracy"])
#train the network

print("training network .......")
#H =model.fit(trainData, testLabels, validation_data =(testData, testLabels), batch_size =128 ,epochs = 20 , verbose =1)
H = model.fit(trainData, trainLabels,
validation_data=(testData, testLabels), batch_size=128, epochs=20, verbose=1)

print("evaluating network....")
predictions =model.predict(testData, batch_size =128)
print(classification_report(testLabels.argmax(axis =1), predictions.argmax(axis=1), target_names=[str(x) for x in le.classes_]))

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 20), H.history[ "loss"], label="train_loss")
plt.plot(np.arange(0, 20), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 20), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, 20), H.history["val_acc"], label=  "val_acc")
plt.title("training loss and accuracy")
plt.xlabel("epochs")
plt.ylabel("loss/accuracy")
plt.legend()
plt.show()


    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

