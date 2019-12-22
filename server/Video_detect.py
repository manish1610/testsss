import cv2
import time
from datetime import date
import numpy as np
from xlwt import Workbook 
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyrebase
from firebase import firebase
from firebase.firebase import FirebaseApplication
import tensorflow as tf
from keras import backend as K
from keras.models import load_model
from yolo_utils import scale_boxes
from yad2k.models.keras_yolo import yolo_head, yolo_boxes_to_corners
from yad2k.models.keras_yolo import yolo_eval
from yad2k.yolo_utils import read_classes, read_anchors, preprocess_webcam_image
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator



row=1
column=1 
today = date.today()
name_file = today.strftime("%B %d, %Y")
strings = time.strftime("%H,%M")
t = strings.split(',')
hour_t=int(t[0])
mini_t=int(t[1])
curr_t=hour_t*60+mini_t
session_time=int(input("Enter the session total time in min"))
exit_time=curr_t+session_time
ap = argparse.ArgumentParser()
ap.add_argument("--mode",help="train/display")
a = ap.parse_args()
mode = a.mode 
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))
model.load_weights('model.h5')
cv2.ocl.setUseOpenCL(False)
emotion_dict = {0: "A", 1: "D", 2: "F", 3: "Ha", 4: "N", 5: "Sa", 6: "Sur"}

def yolo_filter_boxes(box_confidence, boxes, box_class_probs, threshold = .5):

    #Compute box scores
    box_scores = box_confidence*box_class_probs
      
    #the box_classes
    box_classes = K.argmax(box_scores, axis=-1)
    box_class_scores = K.max(box_scores, axis=-1)
    
    
    #Create a filtering mask based on "box_class_scores" 
    filtering_mask = (box_class_scores>threshold)
       
    # Apply the mask to scores, boxes and classes
    scores = tf.boolean_mask(box_class_scores,filtering_mask)

    boxes = tf.boolean_mask(boxes,filtering_mask)
    classes = tf.boolean_mask(box_classes,filtering_mask)
    
    
    return scores, boxes, classes
# yolo_non_max_suppression

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 100,iou_threshold = 0.5):

    
    max_boxes_tensor = K.variable(max_boxes, dtype='int32')     
    K.get_session().run(tf.variables_initializer([max_boxes_tensor])) 
    
    # list of indices corresponding to boxes you keep
    nms_indices = tf.image.non_max_suppression( boxes,scores,max_boxes,iou_threshold)

    
    
    # indices from scores, boxes and classes
    scores = K.gather(scores,nms_indices)
    boxes = K.gather(boxes,nms_indices)
    classes = K.gather(classes,nms_indices)

    
    return scores, boxes, classes

def yolo_eval(yolo_outputs, image_shape = (720., 1280.), max_boxes=100, score_threshold=.5, iou_threshold=0.5):

  
   
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs[0],yolo_outputs[1],yolo_outputs[2],yolo_outputs[3]

    
    boxes = yolo_boxes_to_corners(box_xy, box_wh)
    
    scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, score_threshold)
    

    boxes = scale_boxes(boxes, image_shape)

    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold )
    

    
    return scores, boxes, classes

stream = cv2.VideoCapture(0)
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
image_shape = (720., 1280.)    
yolo_model = load_model("model_data/yolo.h5")
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)
wb = Workbook() 
sheet1 = wb.add_sheet('Sheet 1') 
sheet1.write(0, 0, 'Time')
sheet1.write(0, 1, 'Total_Audience') 
sheet1.write(0, 2, 'Attentive')
sheet1.write(0,3,"Happy_No")
sheet1.write(0,4,"Surprised")
sheet1.write(0,5,"Sad")


def prediction(sess, frame,row):

    # Preprocess
    image, image_data = preprocess_webcam_image(frame, model_image_size=(608, 608))
    out_scores, out_boxes, out_classes =  sess.run([scores, boxes, classes], feed_dict={yolo_model.input: image_data, K.learning_phase(): 0})
    classlen=len(out_classes)
   
    #print(out_classes)
    counter=0
    for i in range(0,classlen):
       if(out_classes[i]==0):
        counter=counter+1

    #print("Total Number of people:", counter)     
    face_cascade=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    count_front=0
    while(1):
     #_, frame=cap.read()
     blk1=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
     face=face_cascade.detectMultiScale(blk1,scaleFactor=1.05,minNeighbors=15)
     break   
    happy=0
    neutral=0
    sad=0  
    for face_co in face:
         x,y,w,h=face_co.reshape(4)
         cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
         cv2.imshow('Detected',frame)
         count_front=len(face)
    for (x, y, w, h) in face:
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = blk1[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            #print(prediction)
            if(prediction[0][3]>0.4):
                happy=happy+1
            if(prediction[0][6]>0.4):
                neutral=neutral+1
            if(prediction[0][5]>0.4):
                sad=sad+1
            maxindex = int(np.argmax(prediction))
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #print("emotion",happy,sad,neutral)
    #print("Total number of attentive people",count_front)   
    strings1 = time.strftime("%H,%M")
    t1 = strings1.split(',')
    hour_t1=int(t1[0])
    mini_t1=int(t1[1])
    curr_t1=hour_t1*60+mini_t1
    exact_t=curr_t1-curr_t
    #print(exact_t,curr_t1,curr_t)
    if(counter<count_front):
        counter=count_front
    sheet1.write(row, column-1, exact_t) 
    sheet1.write(row, column, counter)
    sheet1.write(row, column+1, count_front)
    sheet1.write(row,column+2,happy)
    sheet1.write(row,column+3,neutral)
    sheet1.write(row,column+4,sad)
    row=row+1
    
    #cap.release()
    return np.array(image),row
    
sess = K.get_session()  

while True:
    # Capture frame-by-frame
    grabbed, frame = stream.read()
    if not grabbed:
        break

    # Run detection
    start = time.time()

    output_image,row = prediction(sess, frame,row)
    end = time.time()
    
    wb.save(name_file+".xls")
    time.sleep(30)
    print("processing-------------------->>")
    strings1 = time.strftime("%H,%M")
    t1 = strings1.split(',')
    hour_t1=int(t1[0])
    mini_t1=int(t1[1])
    curr_t1=hour_t1*60+mini_t1
    if(curr_t1==exit_time):
        stream.release()
        cv2.destroyAllWindows()
        break
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
###############################################
        ##Visualization##
dataset=pd.read_excel(name_file+".xls")
data=dataset.copy()
x = np.arange((len(data["Time"])) ) # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, data["Total_Audience"], width, label='Total_Audience')
rects2 = ax.bar(x + width/2, data["Attentive"], width, label='Attentive_Audience')

ax.set_ylabel('Number of Audience')
ax.set_title('Front Face Analysis')
ax.set_xlabel('Time interval(30 sec)')
ax.legend(loc='lower center', bbox_to_anchor=(.3,-0.27), shadow=True, ncol=2)
plt.savefig('bar.png')
plt.clf()

sns.boxplot(data["Time"],data["Attentive"])
plt.savefig("whispbox.png")
plt.clf()


add=sum(data["Attentive"])
add1=sum(data["Total_Audience"])
per=(add/add1)*100
labels = "Attentive","Distracted"
sizes = [per,100-per]
colors = ['gold', 'yellowgreen']
explode = (0.1, 0)  # explode 1st slice
# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.savefig("pie.png")
plt.clf()


total_att=data["Attentive"].sum()
value_c=list(data["Attentive"].value_counts())
bb=data["Attentive"].unique()
label=[]
for i in range(0,len(bb)):
    label.append(str(bb[i])+" audience")
size=[]
for i in range(0,len(value_c)):
    size.append((value_c[i]/total_att)*100)
size.sort(reverse=True)
colors = ['gold', 'yellowgreen',"red","blue","green","violet","purple"]
explode = [0.14] 
for i in range(0,len(size)-1):
    explode.append(0)# explode 1st slice
plt.pie(size, explode=explode,labels=label, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)
plt.axis('equal')
plt.savefig("Atten.png")
plt.clf()


new=[]
for i in range(0,len(data["Sad"])):
    new.append(data["Sad"][i]+data["Happy_No"][i]+data["Surprised"][i])
data["Neutral"]=data["Total_Audience"]-new
size=[]
size.append(data["Happy_No"].sum())
size.append(data["Surprised"].sum())
size.append(data["Neutral"].sum())
size.append(data["Sad"].sum())
names=["Happy","Surprised","Neutral","Sad"]

my_circle=plt.Circle( (0,0), 0.7, color='white')
from palettable.colorbrewer.qualitative import Pastel1_7
plt.pie(size, labels=names, colors=Pastel1_7.hex_colors)
p=plt.gcf()
p.gca().add_artist(my_circle)
plt.savefig("Dogh.png")
plt.clf()

data.plot(x="Time", y=["Total_Audience", "Happy_No", "Surprised","Sad","Neutral"], kind="bar")
ax.set_ylabel('Number of Audience')
ax.set_title('Emotion Analysis')
ax.set_xlabel('Time interval(30 sec)')
ax.legend(loc=-2, bbox_to_anchor=(.3,-0.3), shadow=True, ncol=2)
plt.savefig('bar1.png')
plt.clf()

print("yes")
config = {"apiKey": "AIzaSyCdkWt6qI7cGc5cLxaiAvNWdcK7tr25H8s",
    "authDomain": "administration-62d37.firebaseapp.com",
    "databaseURL": "https://administration-62d37.firebaseio.com",
    "projectId": "administration-62d37",
    "storageBucket": "administration-62d37.appspot.com",
    "messagingSenderId": "480584339659",
    "appId": "1:480584339659:web:fcc5273c441181e1228b4e"}
firebasee= pyrebase.initialize_app(config)
storage=firebasee.storage()
storage.child("bar.png").put("bar.png")
val=storage.child("bar.png").get_url(None)
FBConn= FirebaseApplication('https://administration-62d37.firebaseio.com/', authentication =None)
data_to_upload = {
            'pic1' : val        
           }
result=FBConn.post('pics',data_to_upload)

storage.child("pie.png").put("pie.png")
val=storage.child("pie.png").get_url(None)
FBConn= FirebaseApplication('https://administration-62d37.firebaseio.com/', authentication =None)
data_to_upload = {
            'pic2' : val        
           }
result=FBConn.post('pics',data_to_upload)

storage.child("whispbox.png").put("whispbox.png")
val=storage.child("whispbox.png").get_url(None)
FBConn= FirebaseApplication('https://administration-62d37.firebaseio.com/', authentication =None)
data_to_upload = {
            'pic3' : val        
           }
result=FBConn.post('pics',data_to_upload)
      
storage.child("Atten.png").put("Atten.png")
val=storage.child("Atten.png").get_url(None)
FBConn= FirebaseApplication('https://administration-62d37.firebaseio.com/', authentication =None)
data_to_upload = {
            'pic4' : val        
           }
result=FBConn.post('pics',data_to_upload)

storage.child("bar1.png").put("bar1.png")
val=storage.child("bar1.png").get_url(None)
FBConn= FirebaseApplication('https://administration-62d37.firebaseio.com/', authentication =None)
data_to_upload = {
            'pic5' : val        
           }
result=FBConn.post('pics',data_to_upload)

storage.child("Dogh.png").put("Dogh.png")
val=storage.child("Dogh.png").get_url(None)
FBConn= FirebaseApplication('https://administration-62d37.firebaseio.com/', authentication =None)
data_to_upload = {
            'pic6' : val        
           }
result=FBConn.post('pics',data_to_upload)
print("Done Uploading")




   
    








       
          



