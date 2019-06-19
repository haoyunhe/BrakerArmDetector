# -*- coding:utf-8 -*-
# Stupid python path shit.
# Instead just add darknet.py to somewhere in your python path
# OK actually that might not be a great idea, idk, work in progress
# Use at your own risk. or don't, i don't care

import sys
import cv2
import os
import time
import datetime
import numpy as np
import math

# sys.path.append('./')
from Yolo import Yolo3
from dArmDetector import doubleArmDetector as armDetect
from dColuDetector import doubleColumnDetector as coluDetect


def Caltime(baseTime, sysTime):
    sysTime = datetime.datetime.strftime(sysTime, "%Y/%m/%d")

    baseTime = time.strptime(baseTime, "%Y/%m/%d")
    sysTime = time.strptime(sysTime, "%Y/%m/%d")

    date1 = datetime.datetime(baseTime[0], baseTime[1], baseTime[2])
    date2 = datetime.datetime(sysTime[0], sysTime[1], sysTime[2])
    # print((date2-date1).days)#将天数转成int型
    return (date2 - date1).days


def drawResImg(img, rect, lines, resInfo):
    if rect[0] < 0:
        rect[0] = 0
    if rect[1] < 0:
        rect[1] = 0
    rect_w = rect[2] - rect[0]
    if rect_w >= 400:
        label_w, label_h = 400, 50
        str_loc = (0, 32)
        str_size = 0.75
    elif rect_w >= 300:
        label_w, label_h = 300, 40
        str_loc = (0, 24)
        str_size = 0.56
    elif rect_w >= 200:
        label_w, label_h = 200, 20
        str_loc = (0, 13)
        str_size = 0.36
    elif rect_w >= 100:
        label_w, label_h = 100, 15
        str_loc = (0, 10)
        str_size = 0.19
    else:
        label_w, label_h = rect_w, 15
        str_loc = (0, 10)
        str_size = 0.15
    if resInfo == "OPEN":
        label_w, label_h = 40, 20
        str_loc = (0, 13)
        str_size = 0.35
    label_img = cv2.cvtColor(np.zeros((label_h, label_w), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
    label_img[:, :, 2] = 255
    cv2.putText(label_img, resInfo, str_loc, cv2.FONT_HERSHEY_SIMPLEX, str_size, (0, 0, 0), 1)
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 255), 1)
    ##将label信息覆盖到原图上
    img[rect[1]:rect[1] + label_h, rect[0]:rect[0] + label_w, :] = label_img

    flag = False
    for line in lines:
        for loc in line:
            if math.isnan(loc):
                # print("is nan...")
                flag = True
        if flag:
            flag = False
            continue
        # print("draw.....")
        pt1 = (int(line[0]), int(line[1]))
        pt2 = (int(line[2]), int(line[3]))
        cv2.line(img, pt1, pt2, (255, 0, 0), 2)


def detector():
    curPath = os.getcwd()
    ## init Yolo3
    yolo3 = Yolo3(curPath + "/yolo/libdarknet.so", curPath + "/yolo/yolov3_test.cfg",
                  curPath + "/yolo/yolov3_50000.weights", curPath + "/yolo/braker.data")
    originImg = cv2.imread(sys.argv[1])
    yolo3_res = yolo3.detect_cv(originImg, thresh=0.5)
    if len(yolo3_res) != 0:
        # print("yolo3 res: ", yolo3_res)
        secStageRes = []
        for res in yolo3_res:
            print(res.name)
            if res.name == "DoubleArmFold_IsolatingSwitch_Close":
                tempStatus, tempResLines = armDetect(originImg, res)
                # secStageRes.append(tempRes)
                # print("DoubleArmFold_IsolatingSwitch_Close")
                resStatus = "CLOSE" + tempStatus
                # secStageRes.append(tempResLines)
                drawResImg(originImg, [res.minx, res.miny, res.maxx, res.maxy], tempResLines, resStatus)
            elif res.name == "DoubleArmFold_IsolatingSwitch_Open":
                drawResImg(originImg, [res.minx, res.miny, res.maxx, res.maxy], [], "OPEN")
            elif res.name == "DoubleColumnRotate_IsolatingSwitch_Close":
                tempStatus, tempResLines = coluDetect(originImg, res)
                resStatus = "CLOSE" + tempStatus
                # secStageRes.append(tempResLines)
                print(tempResLines)
                drawResImg(originImg, [res.minx, res.miny, res.maxx, res.maxy], tempResLines, resStatus)
            elif res.name == "DoubleColumnRotate_IsolatingSwitch_Open":
                drawResImg(originImg, [res.minx, res.miny, res.maxx, res.maxy], [], "OPEN")
            else:
                print("no braker....")
        [filePath, fileFullName] = os.path.split(sys.argv[1])
        [fileName, fileFormat] = fileFullName.split(".")
        cv2.imwrite(curPath + "/resImgs/" + fileName + "_result." + fileFormat, originImg)
    else:
        print("yolo detect no braker...")


def detector_path():
    curPath = os.getcwd()
    ## init Yolo3
    yolo3 = Yolo3(curPath + "/yolo/libdarknet.so", curPath + "/yolo/yolov3_test.cfg",
                  curPath + "/yolo/yolov3_50000.weights", curPath + "/yolo/braker.data")
    srcImgs = os.listdir(sys.argv[1])
    num = len(srcImgs)
    for i in range(int(num)):
        print("#####this is " + str(i + 1) + " image: " + srcImgs[i] + "#####")
        [fileName, fileFormat] = srcImgs[i].split(".")
        start = time.time()
        originImg = cv2.imread(sys.argv[1] + srcImgs[i])
        yolo3_res = yolo3.detect_cv(originImg, thresh=0.5)
        if len(yolo3_res) != 0:
            # print("yolo3 res: ", yolo3_res)
            # secStageRes = []
            for res in yolo3_res:
                print(res.name)
                if res.name == "DoubleArmFold_IsolatingSwitch_Close":
                    tempStatus, tempResLines = armDetect(originImg, res)
                    # secStageRes.append(tempRes)
                    # print("DoubleArmFold_IsolatingSwitch_Close")
                    resInfo = "CLOSE" + tempStatus
                    # secStageRes.append(tempResLines)
                    drawResImg(originImg, [res.minx, res.miny, res.maxx, res.maxy], tempResLines, resInfo)
                elif res.name == "DoubleArmFold_IsolatingSwitch_Open":
                    drawResImg(originImg, [res.minx, res.miny, res.maxx, res.maxy], [], "OPEN")
                elif res.name == "DoubleColumnRotate_IsolatingSwitch_Close":
                    tempStatus, tempResLines = coluDetect(originImg, res)
                    resInfo = "CLOSE" + tempStatus
                    # secStageRes.append(tempResLines)
                    drawResImg(originImg, [res.minx, res.miny, res.maxx, res.maxy], tempResLines, resInfo)
                elif res.name == "DoubleColumnRotate_IsolatingSwitch_Open":
                    drawResImg(originImg, [res.minx, res.miny, res.maxx, res.maxy], [], "OPEN")
                else:
                    print("no braker....")
            cv2.imwrite(curPath + "/resImgs/" + fileName + "_result." + fileFormat, originImg)
        else:
            print("yolo detect no braker...")
        end = time.time()
        print('detect an image spend: %s seconds' % (end - start))

# if __name__ == '__main__':
#     # detector()
#     detector_path()
