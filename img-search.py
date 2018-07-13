import cv2
import numpy as np
import time
import os
from threading import Thread



startTime = int(round(time.time()))

template = cv2.imread('template.png')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
succes_list=[]
list = []
list = os.listdir("/home/ffe/Desktop/Image_Search/images")


for i in range(len(list)):
    img_rgb = cv2.imread("images/" + list[i])
    img_gry = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
    tmp = 0

    w, h = template.shape[::-1]
    res = cv2.matchTemplate(img_gry,template,cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)
        tmp +=1

    if (tmp>1):
        cv2.imwrite('result/'+list[i],img_rgb)
        succes_list.append(list[i])
        

endTime = int(round(time.time()))
print("Success: ", succes_list)
print("Time: ", (endTime-startTime) )