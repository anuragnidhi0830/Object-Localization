import numpy as np
import pandas as pd
import cv2
import os
import pickle
import matplotlib.pyplot as plt
path= "F:/flipkart"

train=pd.read_csv(path+"/training.csv")
test=pd.read_csv(path+"/test.csv")

train_1=train.set_index('image_name')
train_name=set()
test_name=set()
for index,row in train.iterrows():
    train_name.add(str(row['image_name']))


mat=train.as_matrix()

train_name=list(sorted(train_name))

X=[]
Y=[]
for name in train_name:
    image=cv2.resize(cv2.imread(path+"/images/"+name),(120,90 ))
    X.append(image)
i=0
for name in train_name:
    Y.append(mat[i,1:])
    i+=1

X=np.asarray(X)
Y=np.asarray(Y)
    
np.save(path+"/train/Y.npy",Y)
np.save(path+"/train/X120.npy",X)

test_X=[]
test=test.set_index('image_name')
for index,row in test.iterrows():
	image=cv2.resize(cv2.imread(path+"/images/"+index),(120,90 ))
	test_X.append(image)
test_X=np.array(test_X)
np.save(path+"/train/test_X120.npy",test_X)

