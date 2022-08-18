import numpy as np
import argparse
import time
import cv2
import os
from keras.preprocessing import image
import matplotlib.pyplot as plt
from IPython.display import Image
from gtts import gTTS
video=cv2.VideoCapture(0)
check,frame=video.read()

print(check)
print(frame)

cv2.imshow("capt",frame)
img1=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
path="C:\\Users\\SAI PRAKASH\\Desktop\\object detection"
cv2.imwrite(os.path.join(path , 'try.jpg'), img1)
#cv2.waitKey(0)
video.release()
cv2.destroyAllWindows()
labelsPath ="C:\\Users\\SAI PRAKASH\\Desktop\\object detection\\coco.txt"
LABELS = open(labelsPath).read().strip().split("\n")
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
	dtype="uint8")
weightsPath = "C:\\Users\SAI PRAKASH\\Desktop\\object detection\\yolov3.weights"
configPath = "C:\\Users\\SAI PRAKASH\\Desktop\\object detection\\yolov3.cfg"
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
pathh ="C:\\Users\\SAI PRAKASH\\Desktop\\try.jpg"
image =cv2.imread(pathh)
(H, W) = image.shape[:2]
for output in layerOutputs:
    for detection in output:
      scores = detection[5:]
      classID = np.argmax(scores)
      confidence = scores[classID]
      if confidence > 0.5:
        box = detection[0:4] * np.array([W, H, W, H])
        (centerX, centerY, width, height) = box.astype("int")
        x = int(centerX - (width / 2))
        y = int(centerY - (height / 2))
        boxes.append([x, y, int(width), int(height)])
        confidences.append(float(confidence))
        classIDs.append(classID)
        idd.append(classID)
        centers.append((centerX, centerY))
        centers.append((centerX, centerY))
idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5,0.3)
if len(idxs) > 0:
    for i in idxs.flatten():
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        color = [int(c) for c in COLORS[classIDs[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
          0.5, color, 2)
        centerX, centerY = centers[i][0], centers[i][1]
        if centerX <= W/3:
            W_pos = "left "
        elif centerX <= (W/3 * 2):
            W_pos = "center "
        else:
            W_pos = "right "
        if centerY <= H/3:
            H_pos = "top "
        elif centerY <= (H/3 * 2):
            H_pos = "mid "
        else:
            H_pos = "bottom "

        texts.append("There is one "+ LABELS[classIDs[i]]+" in "+H_pos + W_pos )
print(texts)
#Image('image.jpg')
cv2.imwrite('C:\\Users\\SAI PRAKASH\\Desktop\\object detection\\output.jpg', image)
display(Image('C:\\Users\\SAI PRAKASH\\Desktop\\onject detection\\output.jpg'))
language = 'en'
#from tempfile import TemporaryFile
#print(texts)
mytext=''
if(len(texts)!=0):
    for ii in range(0,len(texts)):       
        mytext= mytext+" and "+texts[ii]
    print(mytext)
    myobj = gTTS(text=mytext, lang=language, slow=False)
    pat="C:\\Users\\SAI PRAKASH\\Desktop\speech\\obj2"+str(ii)+".mp3"
    myobj.save(pat)
    os.system(pat)
else:
    mytext="No object detected"
    myobj = gTTS(text=mytext, lang=language, slow=False)
    pat="C:\\Users\\SAI PRAKASH\\Desktop\speech\\objn.mp3"
    myobj.save(pat)
    os.system(pat)
