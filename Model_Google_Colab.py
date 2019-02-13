from google.colab import drive
drive.mount('/content/drive/',force_remount=True)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X=np.load('/content/drive/My Drive/Flipkart/X120.npy')
Y=np.load('/content/drive/My Drive/Flipkart/Y.npy')

permutation=np.random.permutation(range(X.shape[0]))
X=X[permutation]
Y=Y[permutation]

from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten,Dropout, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D,Dropout
from keras.models import Model, load_model
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
from keras.regularizers import l2


def basic_net(X, f, filters, stage, block):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters
    X_shortcut = X

    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(F2,(f,f),strides=(1,1),padding='same',name=conv_name_base+'2b',kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3,name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(F3,(1,1),strides=(1,1),padding='valid',name=conv_name_base+'2c',kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3,name=bn_name_base+'2c')(X)

    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
  
    return X

 


def conv_net(X, f, filters, stage, block, s = 2):

    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    F1, F2, F3 = filters

    X_shortcut = X

    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    X = Conv2D(F2,(f,f),strides=(1,1),padding='same',name=conv_name_base+'2b',kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3,name=bn_name_base+'2b')(X)
    X = Activation('relu')(X)

    X = Conv2D(F3,(1,1),strides=(1,1),padding='valid',name=conv_name_base+'2c',kernel_initializer=glorot_uniform())(X)
    X = BatchNormalization(axis=3,name=bn_name_base+'2c')(X)

 
    X_shortcut = Conv2D(F3,(1,1),strides=(s,s),padding='valid',name=conv_name_base+'1',kernel_initializer=glorot_uniform())(X_shortcut)
    X_shortcut = BatchNormalization(axis=3,name=bn_name_base)(X_shortcut)

    X = Add()([X,X_shortcut])
    X = Activation('relu')(X)
   
    return X   


    def  GeekNet(input_shape = (120, 90, 3)):
    
    X_input = Input(input_shape)

    X = ZeroPadding2D((3, 3))(X_input)

    X = Conv2D(64, (7, 7), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform())(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    X = conv_net(X, f = 3, filters = [64, 64, 256], stage = 2, block='a', s = 1)
    X = basic_net(X, 3, [64, 64, 256], stage=2, block='b')
    X = basic_net(X, 3, [64, 64, 256], stage=2, block='c')

    X = conv_net(X,3,[128,128,512],stage=3,block='a',s=2)
    X = basic_net(X,3,[128,128,512],stage=3,block='b')
    X = basic_net(X,3,[128,128,512],stage=3,block='c')
    X = basic_net(X,3,[128,128,512],stage=3,block='d')

    X = conv_net(X,3,[256,256,1024],stage=4,block='a',s=2)
    X = basic_net(X,3,[256,256,1024],stage=4,block='b')
    X = basic_net(X,3,[256,256,1024],stage=4,block='c')
    X = basic_net(X,3,[256,256,1024],stage=4,block='d')
    X = basic_net(X,3,[256,256,1024],stage=4,block='e')
    X = basic_net(X,3,[256,256,1024],stage=4,block='f')

    X = conv_net(X,3,[512,512,2048],stage=5,block='a',s=2)
    X = basic_net(X,3,[512,512,2048],stage=5,block='b')
    X = basic_net(X,3,[512,512,2048],stage=5,block='c')

    X = AveragePooling2D((2,2),name='avg_pool')(X)
 
    X = Flatten()(X)
    X = Dense(1024,name='fc1' , kernel_initializer = glorot_uniform())(X)
    X = Dense(4, name='fc2' , kernel_initializer = glorot_uniform(),kernel_regularizer=l2(0.01))(X)
    
   
    model = Model(inputs = X_input, outputs = X, name='GeekNet')

    return model

    model = GeekNet(input_shape = (90, 120, 3))

opt=Adam(lr=0.000001,decay=0.00002)
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

def iou(y,yh):
  x1=np.maximum(y[:,0],yh[:,0])
  x2=np.minimum(y[:,1],yh[:,1])
  y1=np.maximum(y[:,2],yh[:,2])
  y2=np.minimum(y[:,3],yh[:,3])
  i=np.maximum(x2-x1,0)*np.maximum(y2-y1,0)
  u=(y[:,1]-y[:,0])*(y[:,3]-y[:,2])+(yh[:,1]-yh[:,0])*(yh[:,3]-yh[:,2])-i
  iou=i/u
  return np.sum(iou)/len(iou)


for i in range(50):  
  model.fit(X,Y,epochs=20,batch_size=32,shuffle=True)
  prediction=model.predict(X,batch_size=32,verbose=1)
  accuracy=iou(Y,prediction)
  print(accuracy)


test_X=np.load("/content/drive/My Drive/Flipkart/test_X120.npy")/255
pred=model.predict(test_X,verbose=1,batch_size=32)
test=pd.read_csv('/content/drive/My Drive/Flipkart/test.csv')

for i in range(pred.shape[0]):
  test.loc[i,'x1']=pred[i,0]
  test.loc[i,'x2']=pred[i,1]
  test.loc[i,'y1']=pred[i,2]
  test.loc[i,'y2']=pred[i,3]

test.to_csv('/content/drive/My Drive/Submission.csv',index=False)