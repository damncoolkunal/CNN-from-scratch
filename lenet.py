#!/usr/bin/env python
# coding: utf-8

# In[10]:


#lenet.py implementation
#from tensorflow.keras import Sequential
#from tensorflow.keras.layers import Conv2D, Flatten, Dense
#from keras.models import Sequential
#from keras.layers.convolutional import Conv2D
#from keras.layers.convolutional import MaxPooling2D
#from tensorflow.keras.layers import MaxPooling2D
#from tensorflow.keras.layers.core import Activation
#from keras.layers.core import Flatten
#from keras.layers.core import Dense
#from keras import backend as K

from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.python.keras import activations
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras import backend as K


class LeNet:
    def build(height, width, depth , classes):
        model =Sequential()
        inputShape = (height, width, depth)

        
        
        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)

            
    #we will add layers
        model.add(Conv2D(20, (5, 5), padding="same",input_shape=inputShape))
        model.add(Dense(10, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Dense(10, activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Flatten())
        model.add(Dense(500))
        model.add(Dense(10, activation='relu'))
    
    #softmax classifier
    
        model.add(Dense(classes))
        model.add(Dense(10, activation='softmax'))
    
    #return the extracted model
        return model
        


    
    










