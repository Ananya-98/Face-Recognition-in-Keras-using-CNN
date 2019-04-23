'''created:- April 17th 2019
dataset creation for two users'''



import cv2

cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video width1
cam.set(4, 480) # set video height
face_detector = cv2.CascadeClassifier('Cascades\haarcascade_frontalface_default.xml')
# For each person, enter one numeric face id
face_id = input('\n enter user id end press enter')
print("\n [INFO] Initializing face capture. Look the camera and wait ...")
# Initialize individual sampling face count
count = 0
while(True):
    ret, img = cam.read()
    #img = cv2.flip(img, -1) # flip video image vertically
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray,minNeighbors=5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)     
        count += 1
        if(str(face_id)=='1'):
        # Save the captured image into the datasets folder
            if(count<=200):
                cv2.imwrite("C:\\Users\\anany\\OneDrive\\Desktop\\an_python\\dataset//face_dataset_train//User.1//User." + str(face_id) + '.' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('image', img)
            else:
                cv2.imwrite("C:\\Users\\anany\\OneDrive\\Desktop\\an_python\\dataset//face_dataset_test//User.1//User." + str(face_id) + '.' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('image', img)
            
       
        else:
            
           if(count<=200):
                cv2.imwrite("C:\\Users\\anany\\OneDrive\\Desktop\\an_python\\dataset//face_dataset_train//User.2//User." + str(face_id) + '.' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('image', img)
           else:
                cv2.imwrite("C:\\Users\\anany\\OneDrive\\Desktop\\an_python\\dataset//face_dataset_test//User.2//User." + str(face_id) + '.' +  
                    str(count) + ".jpg", gray[y:y+h,x:x+w])
                cv2.imshow('image', img)
    k = cv2.waitKey(100) # Press 'ESC' for exiting video
    if k == 27:
        break
    elif count >=300: # Take 30 face sample and stop video2
         break
# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()