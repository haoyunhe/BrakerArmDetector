# -*- coding:utf-8 -*-
# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys
import cv2
import time
import datetime

# sys.path.append('./')
from brakerArmDetector import brakerArmDetector
from Yolo import Yolo3

def Caltime(baseTime,sysTime):
    sysTime = datetime.datetime.strftime(sysTime, "%Y/%m/%d")

    baseTime = time.strptime(baseTime, "%Y/%m/%d")
    sysTime = time.strptime(sysTime, "%Y/%m/%d")

    date1=datetime.datetime(baseTime[0],baseTime[1],baseTime[2])
    date2=datetime.datetime(sysTime[0],sysTime[1],sysTime[2])
    # print((date2-date1).days)#将天数转成int型
    return(date2-date1).days

def drawImage(imagePath, yoloRes, armRes):
    img = cv2.imread(imagePath)
    length = len(yoloRes)
    for i in range(length):
        ### draw yolo's result
        if yoloRes[i].minx < 0:
            yoloRes[i].minx = 0
        if yoloRes[i].miny < 0:
            yoloRes[i].miny = 0
        cv2.rectangle(img, (yoloRes[i].minx, yoloRes[i].miny), (yoloRes[i].maxx, yoloRes[i].maxy), (0, 0, 255), 2)
        ### draw arm detector's result
        if len(armRes)!= 0 and len(armRes[i]) != 0:
            closeRatio = armRes[i][0]
            if closeRatio <= 1.5:
                print("braker has been closed.")
                status = "close/("+str(round(closeRatio, 2))+"/1.5)"
                cv2.putText(img, status, (yoloRes[i].minx + 15, yoloRes[i].miny + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
            else:
                print("braker is not closed!!!")
                status = "open/(" + str(round(closeRatio, 2)) + "/1.5)"
                cv2.putText(img, status, (yoloRes[i].minx + 15, yoloRes[i].miny + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1)
            for armLine in armRes[i][1]:
                pt1 = armLine[0]
                pt2 = armLine[1]
                cv2.line(img, pt1, pt2, (255, 0, 0), 2)
        else:
            print("can't detect braker's arm...")
    return img

def detector():
    now_time = datetime.datetime.now()
    base_time = "2018/10/28"
    freeTime = Caltime(base_time, now_time)
    if freeTime > 365:
        print("It's out of date. Please contact NARI's Technical Support Engineer")
    else:
        yolo3 = Yolo3(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        thresh = 0.5
        yolo3_res = yolo3.detect_path(sys.argv[5], thresh)
        if len(yolo3_res) != 0:
            arm_res = brakerArmDetector(sys.argv[5], yolo3_res)
            # print("armRes: ", arm_res)
            # print("armRes len : ", len(arm_res))
            [name, format] = sys.argv[5].split(".")
            img = drawImage(sys.argv[5], yolo3_res, arm_res)
            print("detector result image is "+ name + "_result." + format)
            cv2.imwrite(name + "_result." + format, img)
        else:
            print("yolo detect no braker...")

# if __name__ == '__main__':
#     detector()
