from skimage import io,transform
import glob
import os
import tensorflow as tf 
import numpy as np 
import time
import cv2
import dlib
import sys
from keras.preprocessing import image
from keras.applications import imagenet_utils
from keras.utils import np_utils
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import SGD
from keras.applications.imagenet_utils import decode_predictions
from PIL import Image
import matplotlib.pyplot as plt
import random
from keras.preprocessing import image

ID=('1611463','1611476','1611409',1611471,1611434,1611493,1611436,1611412,1611408,1611419,1611458,1611462,1610763,1611440,1611449,1611438,1611431,
    1611446,1611433,1611418,1611430,1611447,1611490,1611417,1613378,1611453,1611455,1611466,1711459,1611437,1611260,1611420,1611461,1611427,
    1611472,1611482,1611415,1611443,1611407,1611468,1611483,1611460,1611473,1611492,1611426,1611487,1611486,1611450,1611478,1611467,1611444,
    1611413,1611480,1611424,1611488,1613550,1611421,1613376,1611470,1611464,1611465,1611491)

model_path=input("please input the absolute path of modle.h:")
model=load_model(model_path)

def output_id(img):
    #img = image.load_img(img, target_size=(100, 100))
    img = cv2.resize(img, (100,100), interpolation=cv2.INTER_AREA) 
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    features = model.predict_proba(x)
    result = model.predict_classes(x)

    max_features = features[0][0]
    max_label=0
    for i in range(62):
        if features[0][i]>max_features:
            max_features=features[0][i]
            max_label=i
    return ID[max_label]


imglist=[]
reallist=[]
prediclist=[]
randlist=[]
def read_img(path):
    cate=[x for x in os.listdir(path) if os.path.isdir(path+'/'+x)]
    for idx,folder in enumerate(cate):
        rand = random.randint(100,900)
        randlist.append(rand)
        count=100
        for im in glob.glob(path+'/'+folder+'/*.png'):
            if count==rand:
                img=io.imread(im)
                img=transform.resize(img,(100,100,3))
                imglist.append(img)
                reallist.append(folder)
                print "read the picture!!!!!!"
            count=count+1
    return np.asarray(imglist,np.float32)

Y='Y'
N='N'
Is_video=str(input("Do you want to use camera or not [Y]/[N]:"))
if Is_video=='Y':
    detector=dlib.get_frontal_face_detector()
    cap= cv2.VideoCapture(0)
    while True:
        ret , frame= cap.read()

        font=cv2.FONT_HERSHEY_COMPLEX


        if cv2.waitKey(1):
            #cv2.imwrite('/home/huziqi/catkin_ws/src/keras_cnn/video_pictures/now.png',frame)
            #img = cv2.imread("/home/huziqi/catkin_ws/src/keras_cnn/video_pictures/now.png")
            dets=detector(frame,1)
            print("number of faces detected:{}".format(len(dets)))
            for index, face in enumerate(dets):
                print('face {}; left {}; top {}; right {}; bottom {}'.format(index,face.left(), face.top(), face.right(), face.bottom()))
                left = face.left()
                top = face.top()
                right=face.right()
                bottom=face.bottom()
                img=frame[top:bottom,left:right]
                pos_name=tuple([face.left(), int(face.bottom() + (face.bottom() - face.top()) / 4)])
                id_=output_id(img)
                cv2.rectangle(frame,(left,top),(right,bottom),(55,255,155),2)
                cv2.putText(frame, str(id_), pos_name, font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow('frame',frame)
        if(cv2.waitKey(1)==27):
            break
    cap.release()
    cv2.destroyAllWindows()
else:
    is_single=str(input("do you want to test one picture or not [Y]/[N]:"))
    if is_single=='Y':
        path=input("input the absolute path of the picture(please use single quotes):\n")
        src=Image.open(path)
        img_path = path
        img = image.load_img(img_path, target_size=(100, 100))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        label = 5

        features = model.predict_proba(x)
        result = model.predict_classes(x)
        print "////////////////////////////////"
        print "the result of the CNN based on the input image:"
        max_features = features[0][0]
        max_label=0
        for i in range(62):
            if features[0][i]>max_features:
                max_features=features[0][i]
                max_label=i

        print "the school number the person is:"+str(ID[max_label])
        print "the possibility of the result is:"+str(max_features)
        print "////////////////////////////////"


        plt.figure(ID[max_label])
        plt.axis('off')
        plt.imshow(src)
        plt.show()
    
    else:
        path=input("input the absolute path of the database folder(please use single quotes):\n")
        datalist=read_img(path)
        print len(datalist)

        for j in range(62):
            img=datalist[j]
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            
            features = model.predict_proba(x)
            result = model.predict_classes(x)

            max_features = features[0][0]
            max_label=0
            for i in range(62):
                if features[0][i]>max_features:
                    max_features=features[0][i]
                    max_label=i
            prediclist.append(ID[max_label])
            print "the "+str(j)+" predict......."

        acc=0

        for i in range(62):
            if reallist[i]==str(prediclist[i]):
                acc=acc+1
        print "the accuracy is:"+str(acc/62)
        print reallist
        print prediclist
        print randlist

        rrr=random.randint(1,61)
        plt.figure(prediclist[rrr])
        plt.axis('off')
        plt.imshow(datalist[rrr])
        plt.show()


