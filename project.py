from dataclasses import field
from email import header
import os
import numpy as np
import cv2
import face_recognition
from datetime import *     
# ######################################################################
path='faceRecognition\images'     
mylist=os.listdir(path)
# print(mylist)
classNames=[]
images=[]         
for cl in mylist:
    classNames.append(os.path.splitext(cl)[0])
    images.append(cv2.imread(f'{path}/{cl}'))
print(images)
# print(classNames)
# # # # # # ##############################################################
def findencodings(images):
    encodelist=[]
    for img in images:
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode=face_recognition.face_encodings(img)[0]      #[0] is used pick first..first images
        encodelist.append(encode)
    return encodelist
encodelistknownfaces=findencodings(images)
print(encodelistknownfaces)
# # # # # # #############################################################
def markattendence(name):
    with open("faceRecognition\project_result.csv","r+") as f:
        mydatalist=f.readlines()
        # print(mydatalist)
        namelist=[]
        for i in mydatalist:
            entry=i.split(',')
            namelist.append(entry[0])
        if name not in namelist:
            now=datetime.now()
            dt=(now.strftime("%d-%m-%Y"))
            tm=(now.strftime("%I:%M %p"))
            cv2.putText(img,"marking completed.!",(x1+2,y1-15),cv2.FONT_HERSHEY_COMPLEX,.5,(0,0,255),1)
            f.writelines(f'\n{name},{tm},{dt}')
            
           
            
# # # # # # ########################################################
vdo=cv2.VideoCapture(0)
while True:
    x,img=vdo.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)
    imgS=cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)
    
    facescurrframe=face_recognition.face_locations(imgS)
    encodscurrframe=face_recognition.face_encodings(imgS,facescurrframe)
    
    for faceEncodes,faceLoc in zip(encodscurrframe,facescurrframe):
        matches=face_recognition.compare_faces(encodelistknownfaces,faceEncodes)
        facedist=face_recognition.face_distance(encodelistknownfaces,faceEncodes)
        matchindex=np.argmin(facedist)
        name=classNames[np.argmin(facedist)]
        if matches[matchindex]:
            name=classNames[matchindex]
            print(name)
        # print(facedist)
        # print(classNames[np.argmin(facedist)])
        # print(Faceloc)
# # #         ##################################################################################
        y1,x2,y2,x1=faceLoc
        y1,x2,y2,x1=y1*4,x2*4,y2*4,x1*4
        
        markattendence(name)
        now=datetime.now()
        dt=(now.strftime("%d-%m-%Y"))
        tm=(now.strftime("%I:%M %p"))
        cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,0),3)
        cv2.rectangle(img,(x1,y2+3),(x2,y2+27),(255,255,255),-2)
        cv2.putText(img,name,(x1,y2+20),cv2.FONT_HERSHEY_COMPLEX,.6,(0,0,255),1)
        cv2.putText(img,dt,(x1,y2+38),cv2.FONT_HERSHEY_COMPLEX,.4,(0,255,255),1)
        cv2.putText(img,tm,(x1,y2+52),cv2.FONT_HERSHEY_COMPLEX,.4,(0,255,255),1)
       
# #####################################################################################
        
    cv2.imshow('Face',img)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break
      
# # # # # #############################################