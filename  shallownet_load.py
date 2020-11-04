#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Loading a pretrained model from the disc

from imagetoarraypreprocessor import ImageToArrayPreprocessor
from simplepreprocessor import SimplePreprocessor
from simpledatasetoader import SimpleDatasetLoader
from keras.models import load_model
import numpy as np
import argparse
import cv2
from imutils import paths



#parsing the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d" , "--dataset" , required =True , help = "path to the model directory")
ap.add_argument("-m","--model" , required =True,  help = "path to the model file")
args = vars(ap.parse_args())


#defining classes

classLabels  =["Dog" ,"Cat" , "panda"]

print("Information is loading.......")

imagePaths = np.array(lsit(paths.list_images(args["dataset"])))
idxs = np.random.randint(0 , len(imagePaths) , size(10,))
imagePaths = imagePaths[idxs]

#image preprocessors

sp = SimplePreprocessor(32,32)
iap =ImageToArrayPreprocessor()

sdl =SimpleDatasetLoader(preprocessor= [sp, iap])
(data, labels) =sdl.load(imagePaths)
data =data.astype("float")/ 255.0

print("Loading pre trained network....")
model =load_model(args["model"])

print("info is predicting....")
preds =model.predict(data , batch_size =32).argmax(axis =1)

#loop over the smaple images

for (i , images ) in enumerate(imagePaths):
    image =cv2.imread(imagePath)
    cv2.putText(image ,  "Label: {}".format(classLabels[preds[i]]) , (10,30) , cv2.FONT_HERSHEY_SIMPLEX , 0.7 ,
    (0,255,0) ,2)
    
    
    cv2.imshow("Image" , image)
    cv2.waitKey(0)























# In[ ]:




