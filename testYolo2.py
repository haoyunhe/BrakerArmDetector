# -*- coding:utf-8 -*-
# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys
import os
import thread
import time
import cv2

sys.path.append('./')
from brakerArmDetector import brakerArmDetector
from Yolo import Yolo3

yolo3 = Yolo3("./libdarknet.so","/home/wuchao/work/yolo/yolo3/Train/darknet-master/task/GeLiKaiGuan/20181026/yolov3-voc-power_test.cfg", "/home/wuchao/work/yolo/yolo3/Train/darknet-master/task/GeLiKaiGuan/20181026/backup/yolov3-voc-power.backup","/home/wuchao/work/yolo/yolo3/Train/darknet-master/task/GeLiKaiGuan/20181026/power-voc.data")

thresh = 0.5

def drawImage(imagePath,yoloRes,armRes):
    img = cv2.imread(imagePath)
    for res01 in yoloRes:
        # print(res.name)
        if res01.name == "DoubleArmFold_IsolatingSwitch_Close":
            # print(res.minx)
            # print(res.maxx)
            # print(res.miny)
            # print(res.maxy)
            if res01.minx < 0:
                res01.minx = 0
            if res01.miny < 0:
                res01.miny = 0
            cv2.rectangle(img, (res01.minx, res01.miny), (res01.maxx, res01.maxy), (0, 0, 255), 2)

            for res02 in armRes:
                armDistance = res02[0]
                cv2.putText(img, str(int(armDistance)), (res01.minx + 15, res01.miny + 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
                for armLine in res02[1]:
                    pt1 = armLine[0]
                    pt2 = armLine[1]
                    cv2.line(img, pt1, pt2, (255, 0, 0), 2)
    return img


# [('digr', 0.9998231530189514, (543.1868896484375, 466.4017639160156, 1136.0743408203125, 820.8330078125))]
# Darknet
resPath = "/home/wuchao/brakerArmDetector/endtoend/result/"
dirPath = "/home/wuchao/brakerArmDetector/srcImage/"
# dirPath = "/home/wuchao/work/data/GeLiKaiGuan/20181026/JPEGImages/"
files = os.listdir(dirPath)
num = len(files)
for i in range(int(num)):
    print("#####this is " + str(i + 1) + " image: ", files[i], "#####")
    [fileName, fileFormat] = files[i].split(".")
    # img = cv2.imread(dirPath + files[i])
    # rows, cols = img.shape[:2]
    # print("rows: ",rows," cols: ",cols)
    # r = yolo3.detect_cv(img, thresh)
    yolo3_res = yolo3.detect_path(dirPath + files[i], thresh)
    arm_res = brakerArmDetector(dirPath + files[i], yolo3_res)
    if len(arm_res) != 0:
        img = drawImage(dirPath + files[i], yolo3_res, arm_res)
        cv2.imwrite(resPath + fileName + ".jpg", img)
    # cv2.namedWindow("result",0)
    # cv2.resizeWindow("result", 640, 480)
    # cv2.imshow("result", img)
    # cv2.waitKey(0)
