# coding=utf-8
import cv2
import numpy as np
import os
import math
import copy


class Util():
    ###判断两条直线是否平行
    def isParallel(self, line01, line02, thresh=0.05):
        k01 = abs(line01[1] - line01[3]) / abs(line01[0] - line01[2])
        k02 = abs(line02[1] - line02[3]) / abs(line02[0] - line02[2])
        # print("k01: ",k01,"k02: ",k02, "abs(k01-k02): ",abs(k01-k02))
        if abs(k01 - k02) < thresh:
            return True
        else:
            return False

    ### 形态学操作 -- 先膨胀，后腐蚀
    def morphologyImg(self, image, m, n):
        kernel = np.ones((3, 3), np.uint8)
        for i in range(m):
            image = cv2.dilate(image, kernel, iterations=1)
        # cv2.imshow("dilate", image)
        for j in range(n):
            image = cv2.erode(image, kernel, iterations=1)
        # cv2.imshow("erosion", image)
        return image

    def drawLines(self, image, lines, thickness=1):
        for line in lines:
            x1, y1, x2, y2 = line[:]
            # tempK = (y2 - y1) / (x2 - x1)
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), thickness)
            # cv2.circle(image, (int(x1), int(y1)), 5, (0, 0, 255), -1)
            # cv2.circle(image, (int(x2), int(y2)), 5, (0, 255, 0), -1)
            # if tempK >= 0:
            #     cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # else:
            #     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    ###获取两直线之间的角度
    def getCrossAngle(self, line1, line2):
        arr_0 = np.array([(line1[2] - line1[0]), (line1[3] - line1[1])])
        arr_1 = np.array([(line2[2] - line2[0]), (line2[3] - line2[1])])
        cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))  # 注意转成浮点数运算
        return np.arccos(cos_value) * (180 / np.pi)


class HorizontalDetector():
    ###使用Util类
    def __init__(self, util):
        self.util = util

    def filterLines(self, lines, cols, method):
        ##########################################
        ####****第一次过滤：通过区域，直线长度，角度****#
        ##########################################
        def filter01(lines, cols):
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
            return firstResLines

        #####################################################
        ####****第二次拼接：通过首尾是否相接，角度是否相似进行拼接****#
        #####################################################
        def filter02(firstResLines):
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
                    htDistance = math.sqrt(
                        math.pow(headPoint[0] - tailPoint[0], 2) + math.pow(headPoint[1] - tailPoint[1], 2))
                    # print("htDistance: ", htDistance)
                    if htDistance < dThresh:  ### 说明两直线首尾相接
                        # print("aaaaaaaaaaaaaaaaaaaaaa")
                        if True == self.util.isParallel(firstResLines[0], firstResLines[j], 0.05):
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
            return secondResLines

        # print("secondResLines: ", secondResLines)
        #####################################################
        ####***************第三次：确定最终的直线**************#
        #####################################################
        def filter03_line(tempThirdResLines, tempThirdDistance, index):
            tempFline = tempThirdResLines[index][0]
            tempSline = tempThirdResLines[index][1]
            if tempFline[1] >= tempSline[1]:
                line = (tempFline[0], tempFline[1] - tempThirdDistance[0] / 2, tempFline[2],
                        tempFline[3] - tempThirdDistance[0] / 2)
            else:
                line = (tempFline[0], tempFline[1] + tempThirdDistance[0] / 2, tempFline[2],
                        tempFline[3] + tempThirdDistance[0] / 2)
            return line

        def filter03(secondResLines):
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
            # print("tempThirdResLines", tempThirdResLines)
            # print("tempThirdDistance", tempThirdDistance)
            ##### 从中筛选出两对直线
            if len(tempThirdDistance) == 1:
                filter03ResLine = filter03_line(tempThirdResLines, tempThirdDistance, 0)
                thirdResLines.append(filter03ResLine)
            if len(tempThirdDistance) == 2:
                filter03ResLine01 = filter03_line(tempThirdResLines, tempThirdDistance, 0)
                thirdResLines.append(filter03ResLine01)
                filter03ResLine02 = filter03_line(tempThirdResLines, tempThirdDistance, 1)
                thirdResLines.append(filter03ResLine02)
            if len(tempThirdDistance) >= 3:
                copyTempThirdDistance = copy.deepcopy(tempThirdDistance)
                copyTempThirdDistance.sort()
                minD, maxD = min(copyTempThirdDistance[0:3]), max(copyTempThirdDistance[0:3])
                fIndex = tempThirdDistance.index(copyTempThirdDistance[0])
                sIndex = tempThirdDistance.index(copyTempThirdDistance[1])
                if maxD - minD <= 2:
                    tIndex = tempThirdDistance.index(copyTempThirdDistance[2])
                    minIndex, maxIndex = min([fIndex, sIndex, tIndex]), max([fIndex, sIndex, tIndex])

                    filter03ResLine01 = filter03_line(tempThirdResLines, tempThirdDistance, minIndex)
                    thirdResLines.append(filter03ResLine01)
                    filter03ResLine02 = filter03_line(tempThirdResLines, tempThirdDistance, maxIndex)
                    thirdResLines.append(filter03ResLine02)
                else:
                    filter03ResLine01 = filter03_line(tempThirdResLines, tempThirdDistance, fIndex)
                    thirdResLines.append(filter03ResLine01)
                    filter03ResLine02 = filter03_line(tempThirdResLines, tempThirdDistance, sIndex)
                    thirdResLines.append(filter03ResLine02)

            return thirdResLines

        ResLines = []
        if method == 1:
            ResLines = filter01(lines, cols)
        if method == 2:
            firstResLines = filter01(lines, cols)
            ResLines = filter02(firstResLines)
        if method == 3:
            firstResLines = filter01(lines, cols)
            secondResLines = filter02(firstResLines)
            ResLines = filter03(secondResLines)
        # print("reslines: ", ResLines)
        return ResLines

    #### 输入一张彩色图像
    def detector(self, image):
        statusBraker = 0  ###表示开关状态为open
        angle = -1  ###表示未检测出刀闸臂，即无法计算角度值

        lsd = cv2.createLineSegmentDetector(0)
        rows, cols = image.shape[:2]
        # print("rows: ", rows, ", cols:", cols)

        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        grayImg = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        # # cv2.imshow("grayImg", grayImg)
        # grayImg = cv2.Canny(grayImg, 100, 200)
        # cv2.imshow("grayImg", grayImg)
        if rows < 200:
            grayImg = self.util.morphologyImg(grayImg, 1, 1)
        else:
            grayImg = self.util.morphologyImg(grayImg, 5, 4)

        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], np.float32)
        for l in range(1):
            grayImg = cv2.filter2D(grayImg, -1, kernel=kernel)
        # cv2.imshow("ruihua", grayImg)

        # # 中值滤波
        for m in range(3):
            grayImg = cv2.medianBlur(grayImg, 3)
        # cv2.imshow("img_median", grayImg)

        grayImg = cv2.Canny(grayImg, 100, 150)
        # cv2.imshow("grayImg", grayImg)

        # # Detect lines in the image
        resLines = []
        lines = lsd.detect(grayImg)[0]  # Position 0 of the returned tuple are the detected lines
        if lines is None:
            print("no detector line.....")
        else:
            lines1 = lines[:, 0, :]  # 提取为二维
            firstResLines = self.filterLines(lines1, cols, method=2)

            black01 = cv2.cvtColor(np.zeros((rows, cols), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
            self.util.drawLines(black01, firstResLines)
            # cv2.imshow("HHH1", black01)
            black01 = self.util.morphologyImg(black01, 4, 5)

            black01 = cv2.cvtColor(black01, cv2.COLOR_RGB2GRAY)
            # cv2.imshow("HHH", black01)
            lines = lsd.detect(black01)[0]  # Position 0 of the returned tuple are the detected lines
            if lines is None:
                print("no detector line.....")
            else:
                resLines = lines[:, 0, :]  # 提取为二维
                resLines = self.filterLines(resLines, cols, method=3)
            if len(resLines) >= 2:
                angle = self.util.getCrossAngle(resLines[0], resLines[1])
                print("angle: ", angle)
                if angle < 3:
                    statusBraker = 1
                    print("braker is close")
                else:
                    print("braker is open")
            else:
                print("detect braker arm failed...")
        return statusBraker, angle, resLines


class VerticalDetector():
    def __init__(self, util):
        self.util = util

    def decodePatchImg(self, patch):
        rows, cols = patch.shape[:2]
        thresh = round(cols / 10)
        resLoc = []
        if rows * cols != 0:
            for i in range(2, cols - 2):
                if patch[0, i] == 255:
                    temp01 = patch[:, (i - 1):(i + 1)]
                    temp02 = patch[:, i:(i + 2)]
                    if (rows == sum(temp01.ravel() == 255)) or (rows == sum(temp02.ravel() == 255)):
                        if (i < 8 * thresh) and (i > 2 * thresh):
                            resLoc.append(i)
        return resLoc

    def firstFilter(self, imgCode, thresh):
        resCode = []
        tempCode = []
        for i in range(len(imgCode)):
            if i == len(imgCode) - 1:
                if imgCode[i] - imgCode[i - 1] >= thresh:
                    resCode.append(imgCode[i])
                else:
                    tempCode.append(imgCode[i])
                break
            if imgCode[i + 1] - imgCode[i] < thresh:
                tempCode.append(imgCode[i])
            else:
                tempCode.append(imgCode[i])
                if len(tempCode) == 1:
                    resCode.append(tempCode[0])
                elif len(tempCode) == 2:
                    resCode.append(tempCode[0])  ## 2个的，取第一个
                elif len(tempCode) == 3:
                    resCode.append(tempCode[1])  ## 3个的，取中间一个
                else:
                    resCode.append(tempCode[2])  ## 3个以上的，取第三个
                tempCode = []
        if len(tempCode) != 0:
            tempCode.append(imgCode[-1])
            if len(tempCode) == 1:
                resCode.append(tempCode[0])
            elif len(tempCode) == 2:
                resCode.append(tempCode[0])  ## 2个的，取第一个
            elif len(tempCode) == 3:
                resCode.append(tempCode[1])  ## 3个的，取中间一个
            else:
                resCode.append(tempCode[2])  ## 3个以上的，取第三个
        return resCode

    def secondFilter(self, lines):
        #####################################################
        ####****第二次拼接：通过首尾是否相接，角度是否相似进行拼接****#
        #####################################################
        # print(lines)
        firstResLines = np.array(lines)
        # print("firstResLines: ", firstResLines)
        firstResLinesLen = firstResLines.shape[0]
        secondResLines = []
        dThresh = 3
        secondResLinesLen = len(secondResLines)
        # print("firstResLinesLen", firstResLinesLen)
        # print("secondResLinesLen", secondResLinesLen)
        while secondResLinesLen != firstResLinesLen:
            if secondResLinesLen == 0:
                tempLen = firstResLinesLen
            else:
                tempLen = secondResLinesLen
            if secondResLinesLen != 0:
                firstResLinesLen = secondResLinesLen
            # print("tempLen: ", tempLen)
            secondResLines = []
            while tempLen != 0:
                ###1. 对直线根据首点排序
                firstResLines = firstResLines[firstResLines[:, 1].argsort()]  # 按照y1对行排序
                ###2. 寻找首尾相接的直线，设置两点之间的距离阈值
                headPoint = (firstResLines[0, 2], firstResLines[0, 3])  ##取pt2
                tempLines = []
                tempLoc = []
                for j in range(1, tempLen):
                    tailPoint = (firstResLines[j, 0], firstResLines[j, 1])  ##取pt1
                    htDistance = math.sqrt(
                        math.pow(headPoint[0] - tailPoint[0], 2) + math.pow(headPoint[1] - tailPoint[1], 2))
                    # print("htDistance: ", htDistance)
                    if htDistance < dThresh:  ### 说明两直线首尾相接
                        # print("aaaaaaaaaaaaaaaaaaaaaa")
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
                tempLen = firstResLines.shape[0]
            secondResLinesLen = len(secondResLines)
            firstResLines = np.array(secondResLines)
        return secondResLines

    def decodeImage(self, image, bins, thresh01, thresh02):
        ### thresh01: 黑块的宽度；thresh02：黑块中黑色点占的比例
        resPath = "F:/NARI/Data/sichuan/braker/firstDetImages/dstTemp/"
        imageCode = []
        rows, cols = image.shape[:2]
        height = round(rows / bins)
        for i in range(bins):
            begin = i * height
            end = (i + 1) * height
            if end <= rows:
                patchImg = image[begin:end, :]
            else:
                patchImg = image[begin:rows, :]
            # cv2.imwrite(resPath + "patch" + str(i) + ".jpg", patchImg)
            patchImgCode = self.decodePatchImg(patchImg)
            # print("patch" + str(i), "code: ", patchImgCode)
            if len(patchImgCode) > 1:
                patchImgCode = self.firstFilter(patchImgCode, round(cols / 30))
                # print("filter" + str(i), "code: ", patchImgCode)
            if len(patchImgCode) >= 2:
                tempCode = []
                for j in range(len(patchImgCode) - 1):
                    first = patchImgCode[j]
                    second = patchImgCode[j + 1]
                    if first - thresh01 >= 0:
                        tempImg01 = patchImg[:, (first - thresh01):(first - 1)]
                    elif first <= 1:
                        tempImg01 = 255 * np.ones((3, 3), dtype=int)
                    else:
                        tempImg01 = patchImg[:, 0:(first - 1)]

                    if second + thresh01 <= cols:
                        tempImg02 = patchImg[:, (second + 2):(second + thresh01)]
                    elif second + 2 >= cols:
                        tempImg02 = 255 * np.ones((3, 3), dtype=int)
                    else:
                        tempImg02 = patchImg[:, (second + 2):cols]

                    tempImg03 = patchImg[:, (first + 1):second]
                    # print(tempImg03)
                    if sum(tempImg01.ravel() == 0) / (tempImg01.shape[0] * tempImg01.shape[1]) > thresh02 and \
                                            sum(tempImg02.ravel() == 0) / (
                                                tempImg02.shape[0] * tempImg02.shape[1]) > thresh02 and \
                                            sum(tempImg03.ravel() == 0) / (
                                                tempImg03.shape[0] * tempImg03.shape[1]) > thresh02:
                        tempCode.append(first)
                        tempCode.append(second)
                if len(tempCode) != 0:
                    tempCode.insert(0, i)
                    imageCode.append(tempCode)
            if len(patchImgCode) == 1:
                tempCode = []
                first = patchImgCode[0]
                if first - thresh01 * 2 >= 0:
                    tempImg01 = patchImg[:, (first - thresh01 * 2):(first - 1)]
                elif first <= 1:
                    tempImg01 = 255 * np.ones((3, 3), dtype=int)
                else:
                    tempImg01 = patchImg[:, 0:(first - 1)]

                if first + thresh01 * 2 <= cols:
                    tempImg02 = patchImg[:, (first + 2):(first + thresh01 * 2)]
                elif first + 2 >= cols:
                    tempImg02 = 255 * np.ones((3, 3), dtype=int)
                else:
                    tempImg02 = patchImg[:, (first + 2):cols]

                if sum(tempImg01.ravel() == 0) / (tempImg01.shape[0] * tempImg01.shape[1]) > 0.95 and \
                                        sum(tempImg02.ravel() == 0) / (tempImg02.shape[0] * tempImg02.shape[1]) > 0.95:
                    tempCode.append(first)
                if len(tempCode) != 0:
                    tempCode.insert(0, i)
                    imageCode.append(tempCode)

        return imageCode
        # cv2.imwrite(resPath + "patch" + str(i) + ".jpg", patch)

    def detector(self, image):
        statusBraker = 0  ###表示开关状态为open
        angle = -1  ###表示未检测出刀闸臂，即无法计算角度值
        resLines = []
        rows, clos = image.shape[:2]
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        grayImg = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        testEdges = cv2.Canny(grayImg, 100, 200)
        # cv2.imshow("testEdges", testEdges)
        firstCode = self.decodeImage(testEdges, 40, 10, 0.9)
        ###整合成直线信息
        height = round(rows / 40)
        tempLines = []
        for pCode in firstCode:
            tempLines.append((pCode[1], pCode[0] * height, pCode[1], (pCode[0] + 1) * height))

        secResLines = self.secondFilter(tempLines)

        if len(secResLines) >= 2:
            angle = self.util.getCrossAngle(secResLines[0], secResLines[-1])
            resLines.append(secResLines[0])
            resLines.append(secResLines[-1])
            print("angle: ", angle)
            if angle < 5:
                statusBraker = 1
                print("braker is close")
            else:
                print("braker is open")
        else:
            print("detect braker arm failed...")
        return statusBraker, angle, secResLines


def runMain():
    resPath = "F:/NARI/Data/sichuan/braker/firstDetImages/resImage_product/"
    # resPath = "F:/NARI/Data/sichuan/braker/firstDetImages/resImage/"
    # dirPath = "F:/NARI/Data/sichuan/braker/firstDetImages/horizontal/"
    # dirPath = "F:/NARI/Data/sichuan/braker/firstDetImages/vertical/"
    # dirPath = "F:/NARI/Data/sichuan/braker/firstDetImages/srcImage/"
    dirPath = "F:/NARI/Data/sichuan/braker/firstDetImages/scrImage_product/"
    # dirPath = "F:/NARI/Data/DoubleColumnBraker/"
    files = os.listdir(dirPath)
    num = len(files)

    for i in range(int(num)):
        print("#####第" + str(i + 1) + "张图像: ", files[i], "#####")
        [fileName, fileFormat] = files[i].split(".")
        testImg = cv2.imread(dirPath + files[i])
        # testImg = cv2.imread(dirPath + "braker10.jpg")
        rows, cols = testImg.shape[:2]
        utilTool = Util()
        if rows <= cols:
            horDetector = HorizontalDetector(utilTool)
            statusBraker, angle, resLines = horDetector.detector(testImg)
        else:
            verDetector = VerticalDetector(utilTool)
            statusBraker, angle, resLines = verDetector.detector(testImg)
        # black01 = cv2.cvtColor(np.zeros((rows, cols), dtype=np.uint8), cv2.COLOR_GRAY2BGR)
        # utilTool.drawLines(black01, resLines)
        utilTool.drawLines(testImg, resLines, 2)
        cv2.imwrite(resPath + files[i], testImg)
        # black01 = morphologyImg(black01, 4, 5)
        # cv2.imshow("black01", black01)
        # cv2.imshow("testImg", testImg)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


if __name__ == '__main__':
    runMain()
