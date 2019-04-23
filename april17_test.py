'''
created:- april 17th

Testing the xyz
'''

import cv2
from keras.models import model_from_json

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('Cascades\haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
pred = 'unknown'
#count = 0
while(True):
    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,minNeighbors=5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        cv2.putText(img, pred, (x+5,y-5), font, 1, (255,255,255), 2)
        #count += 1
        
        cv2.imwrite("C:\\Users\\anany\\OneDrive\\Desktop\\an_python\\dataset//test_image\\User.jpg", gray[y:y+h,x:x+w] )
        cv2.imshow('image', img)
        
    k = cv2.waitKey(2000) # Press 'ESC' for exiting video
    break
   

print("Image for test captured")
#cam.release()
#cv2.destroyAllWindows()


json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
import numpy as np

from keras.preprocessing import image
test_image = image.load_img("C:\\Users\\anany\\OneDrive\\Desktop\\an_python\\dataset\\test_image\\User.jpg", target_size = (256, 256,3))

test_image = np.expand_dims(test_image, axis = 0)
 
result = loaded_model.predict(test_image)
#x=training_set.class_indices
if (result[0][0] > 0.5):
    prediction = 'Ananya'
elif (result[0][1] > 0.5):
    prediction = 'Anusha'
else:
    prediction ='unknown'
print("The given image is a :-",prediction)


cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('Cascades\haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

#count = 0
while(True):
    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,minNeighbors=5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        cv2.putText(img, prediction, (x+5,y-5), font, 1, (255,255,255), 2)
        #count += 1
        cv2.imshow('image', img)
        
    k = cv2.waitKey(100) # Press 'ESC' for exiting video
    if k == 27:
        break
    '''elif count >= 30: # Take 30 face sample and stop video
         break'''

cam.release()
cv2.destroyAllWindows()