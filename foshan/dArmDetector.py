# -*- coding:utf-8 -*-
import cv2
import numpy as np

"""
  output:[dBraker01,dBraker02....]
         dBraker:[distance,[line01,line02,line02,....]]
         line:[pt1,pt2]
         pt:(x,y)
"""


def doubleArmDetector(img, yolov3_res):
    # secondRes = []
    if yolov3_res.minx < 0:
        yolov3_res.minx = 0
    if yolov3_res.miny < 0:
        yolov3_res.miny = 0
    tempImg = img[yolov3_res.miny:yolov3_res.maxy, yolov3_res.minx:yolov3_res.maxx, :]
    secondResLines = []
    oriImgResStatus = ""
    result = detector(tempImg)

    if len(result) != 0:
        # print(result[0])
        secondResDistance = round(result[0],3)
        oriImgResStatus = " (DISTANCE RATIO: " + str(secondResDistance) + ")"
        for loc in result[1]:
            # pt1 = (loc[0][0] + yolov3_res.minx, loc[0][1] + yolov3_res.miny)
            # pt2 = (loc[1][0] + yolov3_res.minx, loc[1][1] + yolov3_res.miny)
            # tempLines = [pt1, pt2]
            tempLines = (loc[0][0] + yolov3_res.minx, loc[0][1] + yolov3_res.miny,loc[1][0] + yolov3_res.minx, loc[1][1] + yolov3_res.miny)
            secondResLines.append(tempLines)
            # cv2.line(img, pt1, pt2, (255, 0, 0), 2)
        # secondRes.append((secondResDistance, secondResLines))
    else:
        print("double arm detector don't find braker.....")
    return oriImgResStatus, secondResLines


def detector(image):
    result = []
    rows, cols = image.shape[:2]
    blurImg = cv2.GaussianBlur(image, (3, 3), 0)
    grayImg = cv2.cvtColor(blurImg, cv2.COLOR_RGB2GRAY)
    threshold1, threshold2 = adaptiveFindThreshold(grayImg)
    edges = cv2.Canny(grayImg, threshold1, threshold2)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 10, minLineLength=int(rows / 25), maxLineGap=2)
    if not lines is None:
        verticalLines = getVerticalLines(lines, 3, rows, cols)
        # print("verticalLines: ", verticalLines)
        distanceLines = mergeWithDistance(verticalLines, 5)
        # print("distanceLines: ", distanceLines)
        moveBrakerImg, moveBrakerLoc = getMoveBraker(image, distanceLines, 20, int(cols / 5))

        tempMoveBraker = cv2.cvtColor(moveBrakerImg, cv2.COLOR_RGB2GRAY)
        n, bins = np.histogram(tempMoveBraker.ravel(), 256, [0, 256])
        # n, bins, patches = plt.hist(tempMoveBraker.ravel(), 256, [0, 256])
        threshhold = (np.argmax(n[0:128]) + np.argmax(n[128:-1]) + 128) / 2
        tempMoveBraker = np.array(tempMoveBraker).astype(np.uint8)
        ret, tempMoveBraker = cv2.threshold(tempMoveBraker, threshhold + 40, 255, cv2.THRESH_BINARY)

        patchPatterns, patchLocs = getMoveBrakerPatchPatterns(tempMoveBraker, tempMoveBraker.shape[1], 10)
        # print("patchPatterns: ", patchPatterns)
        matchPatterns, matchLocs = matchBrakerArm(patchPatterns, patchLocs)
        # print("matchPatterns: ", matchPatterns)
        if matchPatterns.shape[0] != 0:
            resPatterns, resLocs = procMutilResMatchPatterns(matchPatterns, matchLocs)
            # print("resPatterns: ", resPatterns)
            # print("resLocs", resLocs)
            if resPatterns.shape[0] != 0:
                #print("res patterns: ", resPatterns)
                if len(resPatterns.shape) > 1:
                    brakerInterval = np.mean(resPatterns[:, 5])
                    braker = (np.mean(resPatterns[:, 3]) + np.mean(resPatterns[:, 7])) / 2
                else:
                    brakerInterval = np.mean(resPatterns[5])
                    braker = (np.mean(resPatterns[3]) + np.mean(resPatterns[7])) / 2
                ratio = brakerInterval / braker
                # print("ratio: ", ratio)
                resLocations = getResLocations(moveBrakerLoc, resLocs)
                result.append(ratio)
                result.append(resLocations)
        #     else:
        #         print("no support braker01...............")
        # else:
        #     print("no support braker...............")
    return result


def adaptiveFindThreshold(image):
    highT = 0
    lowT = 0
    percentOfPixelsNotEdges = 0.85
    hist_size = 255

    dx = cv2.Sobel(image, cv2.CV_16S, 1, 0)
    dy = cv2.Sobel(image, cv2.CV_16S, 0, 1)
    dxdy = abs(dx) + abs(dy)
    maxV = np.max(dxdy)
    if maxV == 0:
        return lowT, highT
    if hist_size > maxV:
        hist_size = maxV

    total = (int)(image.shape[0] * image.shape[1] * percentOfPixelsNotEdges)
    sum = 0
    histX, binsX = np.histogram(dxdy.ravel(), hist_size, [0, maxV])
    for i in range(hist_size):
        sum = sum + histX[i]
        if sum > total:
            lowT = ((i + 1) * maxV) / hist_size
            highT = lowT * 3
            break
    return int(lowT), int(highT)


def getVerticalLines(lines, paramTheta, rows, cols):
    colsLen = int(cols / 4)
    rowsLen = int(rows / 6)
    lines1 = lines[:, 0, :]  # 提取为二维
    verticalLines = []
    for x1, y1, x2, y2 in lines1[:]:
        ##过滤，只选取中间部分
        flag = 0
        # if (x1 > colsLen and x1 < 3 * colsLen) and (y1 > 2 * rowsLen and y1 < 5 * rowsLen):
        if (x1 > colsLen and x1 < 3 * colsLen) and (y1 > 2 * rowsLen):
            flag = 1
        if abs(x1 - x2) == 0:
            if flag == 1:
                verticalLines.append((x1, y1, x2, y2))
        else:
            theta = abs(y1 - y2) / abs(x1 - x2)
            if theta > paramTheta and flag == 1:
                verticalLines.append((x1, y1, x2, y2))
    return verticalLines


def mergeWithDistance(lines, distanceThreshhold):
    resLines = []
    while lines:
        # print("lines len: ", len(lines))
        tempDistanceLines = []
        x1 = lines[0][0]
        y1 = lines[0][1]
        x2 = lines[0][2]
        y2 = lines[0][3]
        for i in range(1, len(lines)):
            # print("iiii : ",i)
            tempx1 = lines[i][0]
            tempy1 = lines[i][1]
            tempx2 = lines[i][2]
            tempy2 = lines[i][3]
            meanX = int((x1 + x2) / 2)
            meanTempX = int((tempx1 + tempx2) / 2)
            if (meanTempX < meanX + distanceThreshhold) \
                    and (meanTempX > meanX - distanceThreshhold):
                tempDistanceLines.append((tempx1, tempy1, tempx2, tempy2))

        ## 若不存在距离或角度相近的直线
        if len(tempDistanceLines) == 0:
            resLines.append((x1, y1, x2, y2))
            lines.remove((x1, y1, x2, y2))
        ## 融合距离相近的直线
        else:
            tempDistanceLines.append((x1, y1, x2, y2))
            for i in range(len(tempDistanceLines)):
                lines.remove(tempDistanceLines[i])

            tempDistanceLines = np.array(tempDistanceLines)
            lenY1Y2 = abs(tempDistanceLines[:, 1] - tempDistanceLines[:, 3])
            locLine = np.argmax(lenY1Y2)
            resLines.append((tempDistanceLines[locLine, 0], tempDistanceLines[locLine, 1],
                             tempDistanceLines[locLine, 2], tempDistanceLines[locLine, 3]))

    return resLines


def encodeBrakerArmPattern(image):
    rows, cols = image.shape[:2]
    # print("matchBrakerArm cols: ", cols)
    tempRes = []
    for i in range(cols):
        pixel = image[:, i]
        if np.sum(pixel == 255) > int(3 * rows / 4):
            tempRes.append(1)
        else:
            tempRes.append(0)
    ####
    result = []
    count = 1
    # print("matchBrakerArm tempRes len: ", len(tempRes))
    for j in range(len(tempRes) - 1):
        if tempRes[j] == tempRes[j + 1]:
            count = count + 1
        else:
            result.append((tempRes[j], count))
            count = 1
        if j + 1 == len(tempRes) - 1:
            result.append((tempRes[j + 1], count))
    return result


def mergeBrakerArmPattern(pattern):
    resPattern = []
    while len(pattern) != 0:
        lenPattern = len(pattern)
        if lenPattern >= 3:
            if pattern[0][0] == 1 and pattern[1][0] == 0 and pattern[2][0] == 1:
                if pattern[1][1] <= 5:  ###101(白黑白)模式下，若黑的宽度太小，则融合成白色
                    tempCount = pattern[0][1] + pattern[1][1] + pattern[2][1]
                    # resPattern.append((1,tempCount))
                    for i in range(3):
                        del pattern[0]
                    pattern.insert(0, (1, tempCount))
                else:
                    resPattern.append((pattern[0][0], pattern[0][1]))
                    resPattern.append((pattern[1][0], pattern[1][1]))
                    for i in range(2):
                        del pattern[0]
            else:
                resPattern.append((pattern[0][0], pattern[0][1]))
                del pattern[0]
        else:
            for i in range(lenPattern):
                resPattern.append((pattern[i][0], pattern[i][1]))
            break
    return resPattern


def matchBrakerArm(patterns, locs):
    resPatterns = []
    resLocs = []
    length = len(patterns)
    if length == 0:
        print("no move braker arm.......")
    else:
        for i in range(length):
            pattern = patterns[i]
            loc = locs[i]
            #### 判断是否符合10101模式（即白黑白黑白）
            length2 = len(pattern)
            tempPattern = []
            tempLoc = []
            for j in range(length2):
                if j + 5 <= length2:
                    ### 下面的条件是判断白黑白黑白-10101
                    if (pattern[j][0] == 1) and (pattern[j + 1][0] == 0) and \
                            (pattern[j + 2][0] == 1) and (pattern[j + 3][0] == 0) and (pattern[j + 4][0] == 1):
                        #### 下面的条件是判断两个动触头臂是否近似相等(两者臂宽小于3个像素)
                        if abs(pattern[j + 1][1] - pattern[j + 3][1]) < 3:
                            tempPattern.append((pattern[j][0], pattern[j][1]))
                            tempPattern.append((pattern[j + 1][0], pattern[j + 1][1]))
                            tempPattern.append((pattern[j + 2][0], pattern[j + 2][1]))
                            tempPattern.append((pattern[j + 3][0], pattern[j + 3][1]))
                            tempPattern.append((pattern[j + 4][0], pattern[j + 4][1]))
                            tempLoc0 = pattern[j][1]
                            tempLoc1 = pattern[j][1] + pattern[j + 1][1]
                            tempLoc2 = pattern[j][1] + pattern[j + 1][1] + pattern[j + 2][1]
                            tempLoc3 = pattern[j][1] + pattern[j + 1][1] + pattern[j + 2][1] + pattern[j + 3][1]
                            ### 求位置
                            if j != 0:
                                for k in range(j):
                                    tempLoc0 = tempLoc0 + pattern[k][1]
                                    tempLoc1 = tempLoc1 + pattern[k][1]
                                    tempLoc2 = tempLoc2 + pattern[k][1]
                                    tempLoc3 = tempLoc3 + pattern[k][1]
                                tempLoc.append((loc[0], tempLoc0))
                                tempLoc.append((loc[0], tempLoc1))
                                tempLoc.append((loc[0], tempLoc2))
                                tempLoc.append((loc[0], tempLoc3))
                            else:
                                tempLoc.append((loc[0], tempLoc0))
                                tempLoc.append((loc[0], tempLoc1))
                                tempLoc.append((loc[0], tempLoc2))
                                tempLoc.append((loc[0], tempLoc3))
                        break
            if len(tempPattern) != 0:
                resPatterns.append(np.array(tempPattern).reshape(10))
                resLocs.append(np.array(tempLoc).reshape(8))
    resPatterns = np.array(resPatterns)  #### 匹配的结果：模式数据
    resLocs = np.array(resLocs)  #### 匹配的结果：相对位置信息
    return resPatterns, resLocs


def filterPatternsWithArm(patterns, locs):
    resPatterns = []
    resLocs = []
    length = patterns.shape[0]
    arm = patterns[:, 3]
    n01, bins01 = np.histogram(arm, arm.shape[0], [np.min(arm) - 1, np.max(arm) + 1])
    flag01 = 0
    for i in range(n01.shape[0]):
        if n01[i] >= 2:
            # print("hhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhhh")
            threshhold01 = i
            flag01 = 1
            break
    if flag01 == 0:
        # print("iiiiiiiiiiiiiiiiiiiiiiiiiiii")
        threshhold01 = np.argmax(n01)
    ## arm filter
    flag02 = 0
    for i in range(length):
        if bins01[threshhold01] < patterns[i, 3] and bins01[threshhold01 + 1] > patterns[
            i, 3]:  ### 不同patch之间刀闸臂宽度相差大于5就丢弃
            resPatterns.append(patterns[i])
            resLocs.append(locs[i])
            flag02 = 1

    if flag02 == 0:
        resPatterns = patterns
        resLocs = locs
    else:
        resPatterns = np.array(resPatterns)
        resLocs = np.array(resLocs)

    return resPatterns, resLocs


def filterPatternsWithArmInterval(patterns, locs):
    resPatterns = []
    resLocs = []
    length = patterns.shape[0]
    ## statistical
    armInterval = patterns[:, 5]
    n02, bins02 = np.histogram(armInterval, armInterval.shape[0], [np.min(armInterval) - 1, np.max(armInterval) + 1])
    ## get threshold
    flag03 = 0
    for i in range(n02.shape[0]):
        if n02[i] >= 2:
            threshhold02 = i
            flag03 = 1
            break
    if flag03 == 0:
        threshhold02 = np.argmax(n02)
    ## armInterval filter
    flag04 = 0
    for i in range(length):
        if bins02[threshhold02] < patterns[i, 5] and bins02[threshhold02 + 1] > patterns[i, 5]:
            resPatterns.append(patterns[i])
            resLocs.append(locs[i])
            flag04 = 1

    if flag04 == 0:
        resPatterns = patterns
        resLocs = locs
    else:
        resPatterns = np.array(resPatterns)
        resLocs = np.array(resLocs)

    return resPatterns, resLocs


### 处理matchBrakerArm返回的结果，针对满足10101模式且存在伪刀闸的情况
def procMutilResMatchPatterns(patterns, locs):
    resPatterns = []
    resLocs = []
    length = patterns.shape[0]
    if length == 1:
        resPatterns = patterns
        resLocs = locs
    elif length == 2:
        if abs(patterns[0, 3] - patterns[1, 3]) <= 3:  ##
            resPatterns = patterns
            resLocs = locs
        else:
            resPatterns = patterns[0, :]
            resLocs = locs[0, :]
    else:
        ##
        armPatterns, armLocs = filterPatternsWithArm(patterns, locs)

        ### arm filter failed
        if armPatterns.shape[0] >= 3:
            resPatterns, resLocs = filterPatternsWithArmInterval(armPatterns, armLocs)
        else:
            resPatterns = armPatterns
            resLocs = armLocs
    return resPatterns, resLocs


def getMoveBrakerPatchPatterns(image, w, h):
    # resPath = "F:/NARI/dlWork/houghDetector/shiyan/"
    ### 截取小patch, 高：h, 宽：w
    rows, cols = image.shape[:2]
    iNum = int(rows / h)
    jNum = int(cols / w)
    # if not os.path.exists(resPath + fileName):
    #     os.mkdir(resPath + fileName)
    patchPatterns = []
    patchLocs = []
    for i in range(iNum):
        ibegin = i * h
        iend = (i + 1) * h
        for j in range(jNum):
            jbegin = j * w
            jend = (j + 1) * w
            resImg = image[ibegin:iend, jbegin:jend]
            # print("********************************************************")
            armPattern = encodeBrakerArmPattern(resImg)
            mergePattern = mergeBrakerArmPattern(armPattern)
            if len(mergePattern) >= 5:
                patchPatterns.append(mergePattern)
                patchLocs.append((ibegin, jbegin))
    return patchPatterns, patchLocs


def getMoveBraker(image, lines, hThreshold=10, wThreshold=10):
    if len(lines) == 0:
        imgLoc = (0, 0)
        resImg = image
    else:
        rows, cols = image.shape[:2]
        lines = np.array(lines)
        minX = min(np.min(lines[:, 0]), np.min(lines[:, 2]))
        maxX = max(np.max(lines[:, 0]), np.max(lines[:, 2]))
        minY = min(np.min(lines[:, 1]), np.min(lines[:, 3]))
        maxY = max(np.max(lines[:, 1]), np.max(lines[:, 3]))
        if minX - wThreshold < 0:
            ibegin = 0
        else:
            ibegin = minX - wThreshold
        if maxX + wThreshold > cols:
            iend = cols
        else:
            iend = maxX + wThreshold

        if minY - hThreshold < 0:
            jbegin = 0
        else:
            jbegin = minY - hThreshold
        if maxY + hThreshold > rows:
            jend = rows
        else:
            jend = maxY + hThreshold
        imgLoc = (jbegin, ibegin)
        resImg = image[jbegin:jend, ibegin:iend, :]
    return resImg, imgLoc


def getResLocations(loc1, loc2):
    resLocation = []
    if len(loc2.shape) > 1:
        loc2_len = loc2.shape[0]
        for i in range(loc2_len):
            line1_pt1 = (loc1[1] + loc2[i, 1], loc1[0] + loc2[i, 0])
            line1_pt2 = (loc1[1] + loc2[i, 1], loc1[0] + loc2[i, 0] + 10)
            resLocation.append([line1_pt1, line1_pt2])
            line2_pt1 = (loc1[1] + loc2[i, 3], loc1[0] + loc2[i, 2])
            line2_pt2 = (loc1[1] + loc2[i, 3], loc1[0] + loc2[i, 2] + 10)
            resLocation.append([line2_pt1, line2_pt2])
            line3_pt1 = (loc1[1] + loc2[i, 5], loc1[0] + loc2[i, 4])
            line3_pt2 = (loc1[1] + loc2[i, 5], loc1[0] + loc2[i, 4] + 10)
            resLocation.append([line3_pt1, line3_pt2])
            line4_pt1 = (loc1[1] + loc2[i, 7], loc1[0] + loc2[i, 6])
            line4_pt2 = (loc1[1] + loc2[i, 7], loc1[0] + loc2[i, 6] + 10)
            resLocation.append([line4_pt1, line4_pt2])
    else:
        line1_pt1 = (loc1[1] + loc2[1], loc1[0] + loc2[0])
        line1_pt2 = (loc1[1] + loc2[1], loc1[0] + loc2[0] + 10)
        resLocation.append([line1_pt1, line1_pt2])
        line2_pt1 = (loc1[1] + loc2[3], loc1[0] + loc2[2])
        line2_pt2 = (loc1[1] + loc2[3], loc1[0] + loc2[2] + 10)
        resLocation.append([line2_pt1, line2_pt2])
        line3_pt1 = (loc1[1] + loc2[5], loc1[0] + loc2[4])
        line3_pt2 = (loc1[1] + loc2[5], loc1[0] + loc2[4] + 10)
        resLocation.append([line3_pt1, line3_pt2])
        line4_pt1 = (loc1[1] + loc2[7], loc1[0] + loc2[6])
        line4_pt2 = (loc1[1] + loc2[7], loc1[0] + loc2[6] + 10)
        resLocation.append([line4_pt1, line4_pt2])
    # print("loc2_len: ", loc2_len)
    # print("loc1: ",loc1)
    # print("loc2: ",loc2)

    return resLocation
