# -*- coding: UTF-8 -*-
import numpy as np 
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten
from keras.layers import Conv2D,MaxPooling2D
from keras.optimizers import SGD
from keras.utils import plot_model
from skimage import io,transform
import glob
import os
import time

#读取图片,并将图片resize成100x100
path='/home/huziqi/catkin_ws/Face'
w=100
h=100
c=3
def read_img(path):
    cate=[x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]
    imgs=[]
    labels=[]
    for idx,folder in enumerate(cate):
        for im in glob.glob(path+'/'+folder+'/*.png'):
            print('reading the images:%s'%(im))
            img=io.imread(im)
            img=transform.resize(img,(w,h,c))
            imgs.append(img)
            labels.append(idx)
    return np.asarray(imgs,np.float32),np.asarray(labels,np.int32)
 
data,label=read_img(path)

#打乱顺序
num=data.shape[0]
arr=np.arange(num)
np.random.shuffle(arr)
data=data[arr]
label=label[arr]

#分成训练集和验证集
ratio=0.8
s=np.int(num*ratio)
x_train=data[:s]
y_train=label[:s]
x_test=data[s:]
y_test=label[s:]

y_train = keras.utils.to_categorical(y_train, num_classes=3) 
y_test = keras.utils.to_categorical(y_test, num_classes=3) 

model=Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(100,100,3)))
model.add(Conv2D(32,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.25))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Conv2D(64,(3,3),activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256,activation='relu'))

model.add(Dropout(0.5))

model.add(Dense(3,activation='softmax'))

sgd=SGD(lr=0.01,decay=1e-6,momentum=0.9,nesterov=True)
model.compile(loss='categorical_crossentropy',optimizer=sgd, metrics=['accuracy'])

model.fit(x_train,y_train,batch_size=32,epochs=10)
score=model.evaluate(x_test,y_test,batch_size=32)

plot_model(model,to_file='modelcnn.png',show_shapes=True)
model.save("CNN.h5")
