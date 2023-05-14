import cv2
#from inversek import *
thres = 0.30 # Threshold to detect object
known_distance= 40
tom_width=5
l=[]
c=[]

#cap.set(10,70)

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



def getObjects(img):
    classIds, confs, bbox = net.detect(img, confThreshold=thres)
    #print(classIds,bbox)
    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            if classId == 53:
                cx= int(box[0]+box[2]/2)
                cy= int(box[1]+box[3]/2)
                w=box[2]
                #print(box[0],box[1],box[2],box[3])
                focal_length = 700
                distance = (tom_width * focal_length) / w
                cv2.rectangle(img,box,color=(0,255,0),thickness=2)
                cv2.putText(img,classNames[classId-1].upper(),(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                print(distance,cx,cy)
                for i in range(20):
                    c.append(cx)
                    l.append(distance)
                s1=sum(l)
                s2=sum(c)
                d=s1/20
                x=s2/20
                #print(d,x)
                print(box[0],box[1])
                #inverse(x,d)
                return 1
    return img

   
    
    
if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    cap.set(3,640)
    cap.set(4,480)
    while True:
        success,img = cap.read()
        img=getObjects(img)
        cv2.imshow('Output',img)
        cv2.waitKey(1)
        
    