Introduction
=======
In this project, I make a face_recognition demo based on Keras/Tensorflow. The database I built up based on 62 persons in my class. Each person I collected 1000 face pictures which size 128x128 pixls. Training in CNN, I get a face recognition model of these 62 persons.  
* Accurancy and Loss are show below:
![Image text](result_image/Accuracy.png)  
![Image text](result_image/Loss.png)

* Structure of CNN I used:  
![Image text](result_image/modelcnn.png)

* Three demos of testing:
  * prediction.py: User chooses a face picture(128x128) from 62 test set. And computer will feed back the school ID of the person in that picture, and tell how much is the accurancy rate.
  * prediction_2.py: Program will randomly pick 62 pictures from each person's test set, and make the prediction. Then the computer will feed back the accurancy rate.
  * predictor.py: Program recives the video through computer camera, and mark the face at each frame in real time. Then it will tell the school ID of that person.
