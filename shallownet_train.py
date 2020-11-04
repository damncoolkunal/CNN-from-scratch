#!/usr/bin/env python
# coding: utf-8

# In[2]:


#serializing a model on a disk
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from simpledatasetoader import SimpleDatasetLoader
from shallownet import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder



#parsing the arguments

ap =argparse.ArgumentParser()
ap.add_argument("-d" , "--dataset" , required =True , help ="path to the dataset file")
ap.add_argument("-m" , "--model",  required =True, help ="path to the model file")
args = vars(ap.parse_args())

print("INFO loading images...")
imagePaths =list(paths.list_images(args["dataset"]))

sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()

#load the dataset
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(imagePaths, verbose=500)
data = data.astype("float") / 255.0


(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.02, random_state=42)

#convert the labels from integers to vectors

lb = preprocessing.LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)


print("[INFO] compiling model...")
opt = SGD(lr=0.005)
model =ShallowNet.build(width =32 , height =32 , dpeth =3 , classes =3)

model.compile (loss ="categorical_crossentropy" ,  optimizer ="opt" , metrics =["accuracy"])

#train the network

print("INFO is loading...")

H = model.fit(trainX ,  trainY , validation_data=(testX, testY), batch_size =32 , epochs =100,  verbose =1)

print("INFO serializing network..")
model.save(args["model"])

print("info evaluating net(work..")
predictions =model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
        predictions.argmax(axis=1),
        target_names=["cat", "dog", "panda"]))


#Plotting our loss

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0,100), H.history["loss"], label ="train_loss")
plt.plot(np.arange(0,100), H.history["val_loss"], label ="val_loss")
plt.plot(np.arange(0,100), H.history["acc"], label ="train_acc")
plt.plot(np.arange(0,100), H.history["val_acc"], label = "val_acc")
plt.title("training loss and accuracy")
plt.xlabel("epoch #")
plt.ylabel("loss/accuracy")
plt.legend()
plt.show()




# In[ ]:




