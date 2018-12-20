# coding=utf-8
import cv2
import numpy as np
import os
import math
import copy


def selectResLines(lines):
    lines = np.array(lines)
    lines = lines[lines[:, 0].argsort()]
    firstResLines = []
    ## 第一次选择
    while lines.shape[0] != 0:
        # print(lines.shape[0])
        firstLine = lines[0, :]
        firstLineLen = math.sqrt(
            math.pow(abs(firstLine[0] - firstLine[2]), 2) + math.pow(abs(firstLine[1] - firstLine[3]), 2))
        firstLineK = (firstLine[3] - firstLine[1]) / (firstLine[2] - firstLine[0])

        tempResLines = []
        tempLines = copy.deepcopy(lines)
        tempLoc = []
        flag = 0
        for i in range(1, tempLines.shape[0]):
            tempLine = tempLines[i, :]
            tempLineLen = math.sqrt(
                math.pow(abs(tempLine[0] - tempLine[2]), 2) + math.pow(abs(tempLine[1] - tempLine[3]), 2))
            tempLineK = (tempLine[3] - tempLine[1]) / (tempLine[2] - tempLine[0])
            if abs(tempLine[0] - firstLine[0]) <= 20 and abs(tempLine[1] - firstLine[1]) <= 20:
                if abs(firstLineK - tempLineK) < 0.1:
                    # print("aaaaaaaaaaaaaaaaaaaaaaa")
                    tempResLines.append(tempLine)
                    tempLoc.append(i)
                if i == tempLines.shape[0] - 1:
                    flag = 1
                    # lines = np.delete(lines, i, 0)
            else:
                # print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
                tempResLines.append(firstLine)
                lines = np.delete(lines, 0, 0)
                break
        if flag == 1:
            lines = np.delete(lines, 0, 0)

        lines = np.delete(lines, tempLoc, 0)

        if len(tempResLines) > 1:
            firstResLines.append(tuple(tempResLines))

        if lines.shape[0] == 1:
            break

    ## 第二次选择
    secResLines = []
    if len(firstResLines) > 2:
        loc = np.argsort(-np.array([len(line) for line in firstResLines]))[:2]
        # print("loc: ",loc)
        for tempLoc in loc:
            secTempLines = firstResLines[tempLoc]
            secTempLinesLen = [math.sqrt(
                math.pow(abs(secTempLine[0] - secTempLine[2]), 2) + math.pow(abs(secTempLine[1] - secTempLine[3]), 2))
                               for secTempLine in secTempLines]
            secTempLinesMaxLoc = np.argsort(-np.array(secTempLinesLen))[0]
            secResLines.append(tuple(secTempLines[secTempLinesMaxLoc]))
    elif len(firstResLines) == 2:
        for m in range(2):
            secTempLines = firstResLines[m]
            secTempLinesLen = [math.sqrt(
                math.pow(abs(secTempLine[0] - secTempLine[2]), 2) + math.pow(abs(secTempLine[1] - secTempLine[3]), 2))
                               for secTempLine in secTempLines]
            secTempLinesMaxLoc = np.argsort(-np.array(secTempLinesLen))[0]
            secResLines.append(tuple(secTempLines[secTempLinesMaxLoc]))
    else:
        secResLines = firstResLines
    return secResLines


# ### 在预置直线上方的直线保留
# def filterLines(lines, minLenLine, priInfo):
#     ### 预置信息的直线方程
#     ## k = (priInfo[3]-priInfo[1])/(priInfo[2]-priInfo[0])
#     ## y = k*x + (priInfo[1] - k*priInfo[0] )
#     resLines = []
#     pcol1, prow1, pcol2, prow2 = priInfo[:]  ## 取出预置信息的坐标
#     for x1, y1, x2, y2 in lines[:]:
#         lenLine = math.sqrt(math.pow(abs(x1 - x2), 2) + math.pow(abs(y1 - y2), 2))
#         if lenLine > minLenLine:
#             # print("aaaaaaaaaaaaaaa")
#             if prow1 - prow2 == 0:
#                 # print("bbbbbbbbbbbbbbbbbbbbb")
#                 if y1 <= prow1 and y2 <= prow2 and \
#                                         min(pcol1, pcol2) <= x1 <= max(pcol1, pcol2) and \
#                                         min(pcol1, pcol2) <= x1 <= max(pcol1, pcol2):
#                     # print("ccccccccccccccccccccc")
#                     if x1 < x2:
#                         resLines.append((x1, y1, x2, y2))
#                     else:
#                         resLines.append((x2, y2, x1, y1))
#             else:
#                 # print("ddddddddddddddddddd")
#                 k = (prow2 - prow1) / (pcol2 - pcol1)
#                 yTemp1 = k * x1 + prow1 - k * pcol1
#                 yTemp2 = k * x2 + prow1 - k * pcol1
#                 if y1 <= yTemp1 and y2 <= yTemp2 and \
#                                         min(pcol1, pcol2) <= x1 <= max(pcol1, pcol2) and \
#                                         min(pcol1, pcol2) <= x1 <= max(pcol1, pcol2):
#                     # print("eeeeeeeeeeeeeeeeeeeeeee")
#                     if x1 < x2:
#                         resLines.append((x1, y1, x2, y2))
#                     else:
#                         resLines.append((x2, y2, x1, y1))
#     return resLines
#

def isParallel(line01, line02, thresh=0.05):
    k01 = abs(line01[1] - line01[3]) / abs(line01[0] - line01[2])
    k02 = abs(line02[1] - line02[3]) / abs(line02[0] - line02[2])
    # print("k01: ",k01,"k02: ",k02, "abs(k01-k02): ",abs(k01-k02))
    if abs(k01 - k02) < thresh:
        return True
    else:
        return False


def filterLines(lines, cols, rows, argThresh):
    ##########################################
    ####****第一次过滤：通过区域，直线长度，角度****#
    ##########################################
    firstResLines = []
    #### 设置区域阈值
    begin = 2 * (round(cols / 10))
    end = 8 * (round(cols / 10))
    ### 设置直线长度阈值
    lenThresh = 1 * (round(cols / 7))
    ### 设置角度阈值
    angleThresh = 5
    for x1, y1, x2, y2 in lines[:]:
        minX = min(x1, x2)
        maxX = max(x1, x2)
        lenLine = math.sqrt(math.pow(abs(x1 - x2), 2) + math.pow(abs(y1 - y2), 2))
        if abs(x1 - x2) != 0:
            kLine = abs(y1 - y2) / abs(x1 - x2)
        else:
            kLine = 1000
        if minX < begin or minX > end:  ## not in [begin,end]
            if (lenLine > lenThresh) and (maxX >= begin or maxX <= end):  ### 直线足够长
                if kLine <= angleThresh:
                    if x1 <= x2:
                        firstResLines.append((x1, y1, x2, y2))
                    else:
                        firstResLines.append((x2, y2, x1, y1))
        else:
            if kLine <= angleThresh and lenLine > lenThresh / 2:
                if x1 <= x2:
                    firstResLines.append((x1, y1, x2, y2))
                else:
                    firstResLines.append((x2, y2, x1, y1))

    #####################################################
    ####****第二次拼接：通过首尾是否相接，角度是否相似进行拼接****#
    #####################################################
    firstResLines = np.array(firstResLines)
    # print("firstResLines: ", firstResLines)
    firstResLinesLen = firstResLines.shape[0]
    secondResLines = []
    dThresh = 3
    while firstResLinesLen != 0:
        ###1. 对直线根据首点排序
        firstResLines = firstResLines[firstResLines[:, 0].argsort()]  # 按照第1列对行排序
        ###2. 寻找首尾相接的直线，设置两点之间的距离阈值
        headPoint = (firstResLines[0, 2], firstResLines[0, 3])  ##取pt2
        tempLines = []
        tempLoc = []
        for j in range(1, firstResLinesLen):
            tailPoint = (firstResLines[j, 0], firstResLines[j, 1])  ##取pt1
            htDistance = math.sqrt(math.pow(headPoint[0] - tailPoint[0], 2) + math.pow(headPoint[1] - tailPoint[1], 2))
            # print("htDistance: ", htDistance)
            if htDistance < dThresh:  ### 说明两直线首尾相接
                # print("aaaaaaaaaaaaaaaaaaaaaa")
                if True == isParallel(firstResLines[0], firstResLines[j], 0.05):
                    # print("bbbbbbbbbbbbbbbbbbbbb")
                    tempLines.append(firstResLines[j])
                    tempLoc.append(j)
        tempLinesLen = len(tempLines)
        tempLoc.append(0)
        if tempLinesLen != 0:
            secondResLines.append((firstResLines[0, 0], firstResLines[0, 1], tempLines[tempLinesLen - 1][2],
                                   tempLines[tempLinesLen - 1][3]))
        else:
            secondResLines.append(tuple(firstResLines[0]))
        firstResLines = np.delete(firstResLines, tempLoc, axis=0)
        firstResLinesLen = firstResLines.shape[0]
    # print("secondResLines: ", secondResLines)
    #####################################################
    ####***************第三次：确定最终的直线**************#
    #####################################################
    secondResLines = np.array(secondResLines)
    secondResLinesLen = secondResLines.shape[0]
    thirdResLines = []
    tempThirdResLines = []
    tempThirdDistance = []
    while secondResLinesLen != 0:
        fLine = secondResLines[0]
        # print("*********************")
        tempLines = []
        tempDistance = []
        for j in range(1, secondResLinesLen):
            sLine = secondResLines[j]
            x1Max = max(fLine[0], sLine[0])
            x2Min = min(fLine[2], sLine[2])
            if x2Min > x1Max:  ### 说明两个直线在x方向有重合部分
                x1x2Mean = int((x1Max + x2Min) / 2)
                fLineK = (fLine[3] - fLine[1]) / (fLine[2] - fLine[0])
                sLineK = (sLine[3] - sLine[1]) / (sLine[2] - sLine[0])
                # print("abs(fLineK-sLineK):", abs(fLineK - sLineK))
                if abs(fLineK - sLineK) < 0.1:  ### 说明两直线是否近似平行
                    fLineY01 = fLineK * x1x2Mean + fLine[1] - fLineK * fLine[0]
                    sLineY01 = sLineK * x1x2Mean + sLine[1] - sLineK * sLine[0]
                    distance01 = abs(fLineY01 - sLineY01)
                    # print("distance: ", distance01)
                    tempDistance.append(distance01)
                    tempLines.append([fLine, sLine])
        if len(tempDistance) != 0:
            minDistance = min(tempDistance)
            minIndex = tempDistance.index(minDistance)
            tempThirdResLines.append(tempLines[minIndex])
            tempThirdDistance.append(minDistance)
        secondResLines = np.delete(secondResLines, 0, axis=0)
        secondResLinesLen = secondResLines.shape[0]
    # print("tempThirdResLines",tempThirdResLines)
    # print("tempThirdDistance",tempThirdDistance)
    ##### 从中筛选出两对直线
    if len(tempThirdDistance) == 1:
        tempFline = tempThirdResLines[0][0]
        tempSline = tempThirdResLines[0][1]
        if tempFline[1] >= tempSline[1]:
            thirdResLines.append((tempFline[0], tempFline[1] - tempThirdDistance[0] / 2, tempFline[2],
                                      tempFline[3] - tempThirdDistance[0] / 2))
        else:
            thirdResLines.append((tempFline[0], tempFline[1] + tempThirdDistance[0] / 2, tempFline[2],
                                      tempFline[3] + tempThirdDistance[0] / 2))
    if len(tempThirdDistance) >= 2:
        copyTempThirdDistance = copy.deepcopy(tempThirdDistance)
        copyTempThirdDistance.sort()
        fMinDistanceIndex = tempThirdDistance.index(copyTempThirdDistance[0])
        sMinDistanceIndex = tempThirdDistance.index(copyTempThirdDistance[1])

        tempFline = tempThirdResLines[fMinDistanceIndex][0]
        tempSline = tempThirdResLines[fMinDistanceIndex][1]
        if tempFline[1] >= tempSline[1]:
            thirdResLines.append((tempFline[0], tempFline[1] - tempThirdDistance[0] / 2, tempFline[2],
                                  tempFline[3] - tempThirdDistance[0] / 2))
        else:
            thirdResLines.append((tempFline[0], tempFline[1] + tempThirdDistance[0] / 2, tempFline[2],
                                  tempFline[3] + tempThirdDistance[0] / 2))

        tempFline = tempThirdResLines[sMinDistanceIndex][0]
        tempSline = tempThirdResLines[sMinDistanceIndex][1]
        if tempFline[1] >= tempSline[1]:
            thirdResLines.append((tempFline[0], tempFline[1] - tempThirdDistance[0] / 2, tempFline[2],
                                  tempFline[3] - tempThirdDistance[0] / 2))
        else:
            thirdResLines.append((tempFline[0], tempFline[1] + tempThirdDistance[0] / 2, tempFline[2],
                                  tempFline[3] + tempThirdDistance[0] / 2))

    return thirdResLines


def drawLines(image, liness):
    for lines in liness:
        for line in lines:
            x1, y1, x2, y2 = line[:]
            tempK = (y2 - y1) / (x2 - x1)
            if tempK >= 0:
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 1)
            else:
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                # cv2.circle(image, (x1, y1), 5, (0, 0, 255), -1)
                # cv2.circle(image, (x2, y2), 5, (0, 255, 0), -1)


def drawLines02(image, lines):
    for line in lines:
        x1, y1, x2, y2 = line[:]
        tempK = (y2 - y1) / (x2 - x1)
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 1)
        cv2.circle(image, (int(x1), int(y1)), 5, (0, 0, 255), -1)
        cv2.circle(image, (int(x2), int(y2)), 5, (0, 255, 0), -1)
        # if tempK >= 0:
        #     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # else:
        #     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


def DecodePatchImage(image, bins):
    ### thresh01: 黑块的宽度；thresh02：黑块中黑色点占的比例
    resPath = "F:/NARI/Data/sichuan/braker/firstDetImages/dstTemp/"
    imageCode = []
    rows, cols = image.shape[:2]
    weight = round(cols / bins)
    for i in range(bins):
        begin = i * weight
        end = (i + 1) * weight
        if end <= cols:
            patchImg = image[:, begin:end]
        else:
            patchImg = image[:, begin:rows]
            # cv2.imwrite(resPath + "patch" + str(i) + ".jpg", patchImg)

    return imageCode
    # cv2.imwrite(resPath + "patch" + str(i) + ".jpg", patch)


def binImage(image):
    n, bins = np.histogram(image.ravel(), 256, [0, 256])
    # threshhold = (np.argmax(n[0:128]) + np.argmax(n[128:-1]) + 128) / 2
    threshhold = (np.argmax(n))
    grayImg = np.array(image).astype(np.uint8)
    ret, thresh = cv2.threshold(grayImg, threshhold + 10, 255, cv2.THRESH_BINARY)
    return thresh
    # cv2.imshow("thresh", thresh)


def runMain():
    resPath = "F:/NARI/Data/sichuan/braker/firstDetImages/dstTemp/"
    dirPath = "F:/NARI/Data/sichuan/braker/firstDetImages/horizontal/"
    # dirPath = "F:/NARI/Data/sichuan/braker/firstDetImages/vertical/"
    # dirPath = "F:/NARI/Data/DoubleColumnBraker/"
    files = os.listdir(dirPath)
    num = len(files)
    lsd = cv2.createLineSegmentDetector(0)
    for i in range(int(num)):
        print("#####第" + str(i + 1) + "张图像: ", files[i], "#####")
        [fileName, fileFormat] = files[i].split(".")
        testImg = cv2.imread(dirPath + files[i])
        # testImg = cv2.imread(dirPath + "firstBraker13.jpg")
        rows, cols = testImg.shape[:2]
        print("rows: ", rows, ", cols:", cols)

        blurred = cv2.GaussianBlur(testImg, (3, 3), 0)
        grayImg = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        # cv2.imshow("grayImg", grayImg)

        kernel = np.ones((3, 3), np.uint8)
        for j in range(8):
            grayImg = cv2.dilate(grayImg, kernel, iterations=1)
        # cv2.imshow("dilate", grayImg)
        for k in range(6):
            grayImg = cv2.erode(grayImg, kernel, iterations=1)
        # cv2.imshow("erosion", grayImg)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        for l in range(1):
            grayImg = cv2.filter2D(grayImg, -1, kernel=kernel)
        # cv2.imshow("ruihua", grayImg)

        # # 均值滤波
        # img_mean = cv2.blur(grayImg, (3, 3))
        # cv2.imshow("img_mean", img_mean)
        # # 高斯滤波
        # img_Guassian = cv2.GaussianBlur(grayImg, (3, 3), 0)
        # cv2.imshow("img_Guassian", img_Guassian)
        # # 中值滤波
        grayImg = cv2.medianBlur(grayImg, 3)
        # cv2.imshow("img_median", grayImg)
        # # 双边滤波
        # img_bilater = cv2.bilateralFilter(grayImg, 9, 75, 75)
        # cv2.imshow("img_bilater", img_bilater)

        thresh = binImage(grayImg)
        thresh = cv2.medianBlur(thresh, 3)
        # cv2.imshow("thresh", thresh)
        # cv2.imwrite(resPath + "thresh" + str(i) + ".jpg", thresh)

        # # Detect lines in the image
        black01 = cv2.cvtColor(np.zeros((rows, cols), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
        lines = lsd.detect(thresh)[0]  # Position 0 of the returned tuple are the detected lines
        lines1 = lines[:, 0, :]  # 提取为二维
        # print("origin line len: ", lines1.shape[0])
        # # filterLines01 = filterLines(lines1, int(cols / 20), (0, rows, cols, rows))
        filterLines01 = filterLines(lines1, cols, rows, 5)
        drawLines02(black01, filterLines01)
        drawLines02(testImg, filterLines01)
        cv2.imshow("black01", black01)

        cv2.imshow("testImg", testImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    runMain()
