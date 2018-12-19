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
            firstResLines.append(tempResLines)

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
            secResLines.append(secTempLines[secTempLinesMaxLoc])
    elif len(firstResLines) == 2:
        for m in range(2):
            secTempLines = firstResLines[m]
            secTempLinesLen = [math.sqrt(
                math.pow(abs(secTempLine[0] - secTempLine[2]), 2) + math.pow(abs(secTempLine[1] - secTempLine[3]), 2))
                               for secTempLine in secTempLines]
            secTempLinesMaxLoc = np.argsort(-np.array(secTempLinesLen))[0]
            secResLines.append(secTempLines[secTempLinesMaxLoc])
    else:
        secResLines = firstResLines

    return secResLines


### 在预置直线上方的直线保留
def filterLines(lines, minLenLine, priInfo):
    ### 预置信息的直线方程
    ## k = (priInfo[3]-priInfo[1])/(priInfo[2]-priInfo[0])
    ## y = k*x + (priInfo[1] - k*priInfo[0] )
    resLines = []
    pcol1, prow1, pcol2, prow2 = priInfo[:]  ## 取出预置信息的坐标
    for x1, y1, x2, y2 in lines[:]:
        lenLine = math.sqrt(math.pow(abs(x1 - x2), 2) + math.pow(abs(y1 - y2), 2))
        if lenLine > minLenLine:
            # print("aaaaaaaaaaaaaaa")
            if prow1 - prow2 == 0:
                # print("bbbbbbbbbbbbbbbbbbbbb")
                if y1 <= prow1 and y2 <= prow2 and \
                                        min(pcol1, pcol2) <= x1 <= max(pcol1, pcol2) and \
                                        min(pcol1, pcol2) <= x1 <= max(pcol1, pcol2):
                    # print("ccccccccccccccccccccc")
                    if x1 < x2:
                        resLines.append((x1, y1, x2, y2))
                    else:
                        resLines.append((x2, y2, x1, y1))
            else:
                # print("ddddddddddddddddddd")
                k = (prow2 - prow1) / (pcol2 - pcol1)
                yTemp1 = k * x1 + prow1 - k * pcol1
                yTemp2 = k * x2 + prow1 - k * pcol1
                if y1 <= yTemp1 and y2 <= yTemp2 and \
                                        min(pcol1, pcol2) <= x1 <= max(pcol1, pcol2) and \
                                        min(pcol1, pcol2) <= x1 <= max(pcol1, pcol2):
                    # print("eeeeeeeeeeeeeeeeeeeeeee")
                    if x1 < x2:
                        resLines.append((x1, y1, x2, y2))
                    else:
                        resLines.append((x2, y2, x1, y1))
    return resLines


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
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # if tempK >= 0:
        #     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        # else:
        #     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


def runMain():
    resPath = "F:/NARI/Data/sichuan/braker/firstDetImages/dst/"
    dirPath = "F:/NARI/Data/sichuan/braker/firstDetImages/horizontal/"
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
        testEdges = cv2.Canny(grayImg, 100, 200)
        cv2.imshow("testEdges", testEdges)

        # n, bins = np.histogram(grayImg.ravel(), 256, [0, 256])
        # threshhold = (np.argmax(n[0:128]) + np.argmax(n[128:-1]) + 128) / 2
        # grayImg = np.array(grayImg).astype(np.uint8)
        # ret, binaryImg = cv2.threshold(grayImg, threshhold + 50, 255, cv2.THRESH_BINARY)

        # Detect lines in the image
        black01 = cv2.cvtColor(np.zeros((rows, cols), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
        lines = lsd.detect(testEdges)[0]  # Position 0 of the returned tuple are the detected lines
        lines1 = lines[:, 0, :]  # 提取为二维
        # print("origin line len: ", lines1.shape[0])
        filterLines01 = filterLines(lines1, int(cols / 20), (0, rows, cols, rows))
        # print("filter line len: ", len(filterLines01))
        resLines = selectResLines(filterLines01)
        # print("resLines len: ", len(resLines))

        if len(resLines) != 0:
            drawLines02(black01, filterLines01)
            drawLines02(testImg, resLines)
            # cv2.line(black01, (68, 90+20),(316, 210+20), (255, 255, 255), 2)
            # cv2.line(binaryImg, (68, 90+20),(316, 210+20), (255, 255, 255), 2)
        else:
            print("no detect lines......")

        # cv2.imwrite(resPath + fileName + "_result.jpg", testImg)
        cv2.imshow("black01", black01)
        cv2.imshow("testImg", testImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    runMain()
