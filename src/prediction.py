from keras.preprocessing import image
from keras.applications import imagenet_utils
import numpy as np
from keras.utils import np_utils
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import SGD
from keras.applications.imagenet_utils import decode_predictions
from PIL import Image
import matplotlib.pyplot as plt
 
model = load_model('All_CNN.h5')

ID=(1611463,1611476,1611409,1611471,1611434,1611493,1611436,1611412,1611408,1611419,1611458,1611462,1610763,1611440,1611449,1611438,1611431,
    1611446,1611433,1611418,1611430,1611447,1611490,1611417,1613378,1611453,1611455,1611466,1711459,1611437,1611260,1611420,1611461,1611427,
    1611472,1611482,1611415,1611443,1611407,1611468,1611483,1611460,1611473,1611492,1611426,1611487,1611486,1611450,1611478,1611467,1611444,
    1611413,1611480,1611424,1611488,1613550,1611421,1613376,1611470,1611464,1611465,1611491)

src=Image.open('/home/huziqi/catkin_ws/src/face_recognition/prediction/10003.png')
img_path = '/home/huziqi/catkin_ws/src/face_recognition/prediction/10003.png'
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