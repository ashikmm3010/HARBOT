import cv2
from inversek import *
from Stepper import *
thres = 0.30 # Threshold to detect object
known_distance= 40
tom_width=5
l=[]
c=[]


def passval(c,l):
    l1=len(c)
    l2=len(l)
    s1=sum(c)
    s2=sum(l)
    x=int((s1/l1)*0.05+2-10)
    d=int(s2/l2+2.5)
    print('res',x,d)
    inverse(x,d)
 
 
classNames= []
classFile = '/home/admin/Documents/Object Detection/coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

#print(classNames)
configPath = "/home/admin/Documents/Object Detection/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/admin/Documents/Object Detection/frozen_inference_graph.pb"

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)



def getObjects(img,k):
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    #print(classIds,bbox)
    p=0
    if len(classIds) != 0: 
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            if classId == 53:
                p=1
                #print('ID')
                cx= int(box[0]+box[2]/2)
                cy= int(box[1]+box[3]/2)
                w=box[2]
                #print(box[0],box[1],box[2],box[3])
                focal_length = 700
                distance = (tom_width * focal_length) / w
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                print(distance,cx*0.05)
                c.append(cx)
                l.append(distance)
                k=k+1
                if k==29:
                    print(k)
                    passval(c,l)
            
            
    return img,p,k

   
    
    
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    k=0
    a=0
    while k!=30:
        success,img = cap.read()
        img,p,k=getObjects(img,k)
        cv2.imshow('Output',img)
        cv2.waitKey(1)
        if p==0:
            cap.release()
            if a<4800:
                TOP('clk',800)
                a=a+800
                cap = cv2.VideoCapture(0)
                cap.set(3,640)
                cap.set(4,480)
            if a>=4800:
                TOP('ant',4800)
                a=0
        
    
