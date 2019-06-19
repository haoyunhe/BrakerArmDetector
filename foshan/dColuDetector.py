# coding=utf-8
import cv2
import numpy as np
import os
import math


class LineTools():
    ### 直线角度
    @classmethod
    def angleOfLine(cls, line):
        if line[0] - line[2] == 0:
            angle = 90
        else:
            k01 = (line[1] - line[3]) / (line[0] - line[2])
            angle = abs(180 * math.atan(k01) / 3.1415)
            # angle = 180 * math.atan(k01) / 3.1415
            # if angle < 0:
            #     print("angle: ", angle)
            #     print(line)
        return angle

    ###判断两条直线是否平行
    @classmethod
    def isParallel(cls, line01, line02, thresh=5):
        angle01 = cls.angleOfLine(line01)
        angle02 = cls.angleOfLine(line02)
        # print("angle01:", angle01, "angle02:", angle02)
        if abs(angle01 - angle02) < thresh:
            return 0
        else:
            return 1

    @classmethod
    def drawLines(cls, image, lines, thickness=1):
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

    @classmethod
    def lengthOfLine(cls, line):
        x1, y1, x2, y2 = line[:]
        lenLine = math.sqrt(math.pow(abs(x2 - x1), 2) + math.pow(abs(y2 - y1), 2))
        return lenLine

    ###获取两直线之间的角度
    @classmethod
    def crossAngleOfLines(cls, line1, line2):
        arr_0 = np.array([(line1[2] - line1[0]), (line1[3] - line1[1])])
        arr_1 = np.array([(line2[2] - line2[0]), (line2[3] - line2[1])])
        cos_value = (float(arr_0.dot(arr_1)) / (np.sqrt(arr_0.dot(arr_0)) * np.sqrt(arr_1.dot(arr_1))))  # 注意转成浮点数运算
        angle = np.arccos(cos_value) * (180 / np.pi)
        if 180 - angle < 10:
            angle = 180 - angle
        return angle

    ### 直线间的距离
    @classmethod
    def distanceOflines(cls, line1, line2):
        A = line2[1] - line2[3]
        B = line2[2] - line2[0]
        C = line2[0] * line2[3] - line2[1] * line2[2]
        distance01 = abs(A * line1[0] + B * line1[1] + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2))
        distance02 = abs(A * line1[2] + B * line1[3] + C) / math.sqrt(math.pow(A, 2) + math.pow(B, 2))
        distance = int((distance01 + distance02) / 2)
        return distance

    ###根据角度和点获取线
    @classmethod
    def produceLine(cls, angle, point, flag, lenLine):
        left, right = lenLine / 5, lenLine
        if flag == "pos":
            pt1 = (point[0] - left, point[1] - abs(math.tan(angle * math.pi / 180)) * left)
            pt2 = (point[0] - right, point[1] - abs(math.tan(angle * math.pi / 180)) * right)
            pt3 = (point[0] + left, point[1] + abs(math.tan(angle * math.pi / 180)) * left)
            pt4 = (point[0] + right, point[1] + abs(math.tan(angle * math.pi / 180)) * right)
            # line1 = (pt1[0], pt1[1], pt2[0], pt2[1])
            # line2 = (pt3[0], pt3[1], pt4[0], pt4[1])
        else:
            pt1 = (point[0] + left, point[1] - abs(math.tan(angle * math.pi / 180)) * left)
            pt2 = (point[0] + right, point[1] - abs(math.tan(angle * math.pi / 180)) * right)
            pt3 = (point[0] - left, point[1] + abs(math.tan(angle * math.pi / 180)) * left)
            pt4 = (point[0] - right, point[1] + abs(math.tan(angle * math.pi / 180)) * right)
        line1 = (pt1[0], pt1[1], pt2[0], pt2[1])
        line2 = (pt3[0], pt3[1], pt4[0], pt4[1])
        return line1, line2


class PointTools():
    ###两点之间的距离
    @classmethod
    def distanceOfPonits(cls, pt1, pt2):
        distance = math.sqrt(math.pow(abs(pt1[0] - pt2[0]), 2) + math.pow(abs(pt1[1] - pt2[1]), 2))
        return distance

    ### 点到直线的距离公式
    @classmethod
    def distOfPtLine(cls, pt, line):
        QP = np.array([pt[0] - line[2], pt[1] - line[3]])
        v = np.array([line[0] - line[2], line[1] - line[3]])
        dist = np.linalg.norm(np.cross(QP, v) / np.linalg.norm(v))
        return dist


class CircleTools():
    ###
    @classmethod
    def relationOfCircles(cls, center1, radius1, center2, radius2):
        ###圆心距
        dist = PointTools.distanceOfPonits(center1, center2)
        if dist > (radius1 + radius2):
            relation = "waili"
        elif dist == (radius1 + radius2):
            relation = "waiqie"
        elif dist == abs(radius1 - radius2):
            relation = "neiqie"
        elif dist < abs(radius1 - radius2):
            relation = "neihan"
        elif dist < (radius1 + radius2):
            relation = "xiangjiao"
        else:
            relation = "error"
        return relation

    @classmethod
    def circumscribedRectangle(cls, center, radius):
        rectOfPt1 = (center[0] - radius, center[1] - radius)
        rectOfPt2 = (center[0] + radius, center[1] + radius)
        rect = [rectOfPt1, rectOfPt2]
        return rect

    @classmethod
    def relationOfLineCircle(cls, line, center, radius):
        dist = PointTools.distOfPtLine(center, line)
        if dist > radius:
            relation = "xiangli"
        elif dist == radius:
            relation = "xiangqie"
        else:
            relation = "xiangjiao"
        return relation

    @classmethod
    def relationOfPointCircle(cls, pt, center, radius):
        disCenter = PointTools.distanceOfPonits(pt, center)
        if disCenter <= radius:
            relation = "yuannei"
        else:
            relation = "yuanwai"
        return relation


class Detector():
    def filteLineWithLength(self, lines, thresh):
        resLines = []
        for line in lines:
            if LineTools.lengthOfLine(line) > thresh:
                resLines.append(tuple(line))
        return resLines

    def filteLineWithCircle(self, lines, center, radius):
        resLines = []
        for line in lines:
            relation1 = CircleTools.relationOfLineCircle(line, center, radius)
            if relation1 == "xiangjiao" or relation1 == "xaingqie":
                relation2 = CircleTools.relationOfPointCircle((line[0], line[1]), center, radius)
                relation3 = CircleTools.relationOfPointCircle((line[2], line[3]), center, radius)
                if relation2 == "yuannei" or relation3 == "yuannei":
                    resLines.append(tuple(line))
        return resLines

    def filteCircles(self, circles, center, radius):
        circles = np.uint16(np.around(circles))
        resCircles = []
        for cir in circles[0, :]:
            relationOfCir = CircleTools.relationOfCircles((cir[0], cir[1]), cir[2], center, radius)
            if relationOfCir == "neiqie" or relationOfCir == "neihan":
                resCircles.append((cir[0], cir[1], cir[2]))
                # draw the outer circle
                # cv2.circle(testImg, (cir[0], cir[1]), cir[2], (0, 255, 0), 2)
                # cv2.circle(black01, (cir[0], cir[1]), cir[2], (0, 255, 0), 2)
        return resCircles

    def fiteWithAngle(self, lines, row, col):
        resLines = []
        ###求直线的角度值
        angles = []
        if row >= col:
            for line in lines:
                angle = LineTools.angleOfLine(line)
                if angle >= 45:
                    angles.append(angle)
        else:
            for line in lines:
                angle = LineTools.angleOfLine(line)
                if angle <= 45:
                    angles.append(angle)
        # print("angles is: ", angles)
        n, bins = np.histogram(angles, 30, [0, 180])
        maxIndex = np.argmax(n)
        angle1, angle2 = bins[maxIndex], bins[maxIndex + 1]

        for line in lines:
            angle = LineTools.angleOfLine(line)
            if angle > angle1 and angle < angle2:
                resLines.append(tuple(line))

        return resLines

    def resLines(self, lines, center, lenLine):
        resLines = []
        angles = []
        pos, neg = 0, 0
        for line in lines:
            ##判断是什么方向的
            if line[0] >= line[2]:
                if line[1] >= line[3]:
                    pos = pos + 1
                else:
                    neg = neg + 1
            else:
                if line[1] >= line[3]:
                    neg = neg + 1
                else:
                    pos = pos + 1
            ##收集角度值
            angle = LineTools.angleOfLine(line)
            angles.append(angle)
        meanAngle = np.mean(angles)
        medianAngle = np.median(angles)
        # print("meanAngle: ", meanAngle, "medianAngle: ", medianAngle)
        angleDiff = abs(meanAngle - medianAngle)
        angleDiff = round(angleDiff, 3)
        # print("angle diff: ", abs(meanAngle - medianAngle))
        if pos > neg:
            # print("this is pos")
            line1, line2 = LineTools.produceLine(medianAngle, center, "pos", lenLine)
        else:
            # print("this is neg")
            line1, line2 = LineTools.produceLine(medianAngle, center, "neg", lenLine)
        resLines.append(line1)
        resLines.append(line2)
        return resLines,angleDiff


def originResLines(lines, yolov3_res):
    resLines = []
    for line in lines:
        x1, y1, x2, y2 = line[:]
        resLines.append((x1 + yolov3_res.minx, y1 + yolov3_res.miny, x2 + yolov3_res.minx, y2 + yolov3_res.miny))
    return resLines


def doubleColumnDetector(img, yolov3_res):
    if yolov3_res.minx < 0:
        yolov3_res.minx = 0
    if yolov3_res.miny < 0:
        yolov3_res.miny = 0
    testImg = img[yolov3_res.miny:yolov3_res.maxy, yolov3_res.minx:yolov3_res.maxx, :]
    rows, cols = testImg.shape[:2]
    CENTER = (int(cols / 2), int(rows / 2))
    if cols >= rows:
        RADIUS = int(cols / 15)
    else:
        RADIUS = int(rows / 15)
    grayImg = cv2.cvtColor(testImg, cv2.COLOR_RGB2GRAY)
    cannyImg = cv2.Canny(grayImg, 100, 130)

    detector = Detector()
    circles = cv2.HoughCircles(cannyImg, cv2.HOUGH_GRADIENT, 5, 10,
                               param1=60, param2=5, minRadius=1, maxRadius=10)
    ###过滤出在预置圆内的圆
    resCircles = detector.filteCircles(circles, CENTER, RADIUS * 2)
    meanCenter = [0, 0]
    if len(resCircles) != 0:
        resCircles = np.array(resCircles)
        centerCircles = resCircles[:, :2]
        meanCenter = np.mean(centerCircles, axis=0)
        # cv2.circle(testImg, (int(meanCenter[0]), int(meanCenter[1])), 5, (255, 255, 0), 2)
    else:
        print("no detect circles...")

    ### 直线检测
    lsd = cv2.createLineSegmentDetector(0)
    lines = lsd.detect(cannyImg)[0]
    oriImgResLines = []
    angleDiff = 90
    if lines is None:
        print("no detector line.....")
    else:
        lines1 = lines[:, 0, :]  # 提取为二维
        lenLines = detector.filteLineWithLength(lines1, RADIUS)
        ###根据预置圆过滤
        cirLines = detector.filteLineWithCircle(lenLines, CENTER, RADIUS * 5)
        # print("cirLines num: ", len(cirLines))
        angleLines = detector.fiteWithAngle(cirLines, rows, cols)
        # print("angleLines num: ", len(angleLines))
        # LineTools.drawLines(testImg, angleLines)
        testImgResLines, angleDiff = detector.resLines(angleLines, meanCenter, RADIUS * 5)
        # LineTools.drawLines(testImg, testImgResLines)
        oriImgResLines = originResLines(testImgResLines, yolov3_res)
    oriImgResStatus = " (ANGLE: " + str(angleDiff) +")"
    return oriImgResStatus, oriImgResLines


def runMain():
    resPath = "D:/NARI/data/DoubleColumnBraker/resImgs/"
    dirPath = "D:/NARI/data/DoubleColumnBraker/srcImgs/"
    files = os.listdir(dirPath)
    num = len(files)
    for i in range(int(num)):
        print("#####第" + str(i + 1) + "张图像: ", files[i], "#####")
        [fileName, fileFormat] = files[i].split(".")
        testImg = cv2.imread(dirPath + files[i])
        # testImg = cv2.imread(dirPath + "DoubleColumnBraker12.jpg", True)
        rows, cols = testImg.shape[:2]
        CENTER = (int(cols / 2), int(rows / 2))
        if cols >= rows:
            RADIUS = int(cols / 15)
        else:
            RADIUS = int(rows / 15)

        # ### flood
        # seed_pt = 212, 132
        # connectivity = 4
        # fixed_range = True
        # mask = np.zeros((rows + 2, cols + 2), np.uint8)
        # mask[:] = 0  # 掩码初始为全0
        # lo = 93  # 观察点像素邻域负差最大值（也就是与选定像素多少差值内的归为同一区域）
        # hi = 70  # 观察点像素邻域正差最大值
        # flags = connectivity  # 低位比特包含连通值, 4 (缺省) 或 8
        # if fixed_range:
        #     flags |= cv2.FLOODFILL_FIXED_RANGE  # 考虑当前象素与种子象素之间的差（高比特也可以为0）
        #     # flags |= cv2.FLOODFILL_MASK_ONLY  # 考虑当前象素与种子象素之间的差（高比特也可以为0）
        # print("flags: ", flags)
        # # 以白色进行漫水填充
        # cv2.floodFill(testImg, mask, seed_pt, (0, 0, 0),
        #               (lo,) * 3, (hi,) * 3, flags)
        # # cv2.circle(testImg, seed_pt, 2, (0, 0, 255), -1)

        grayImg = cv2.cvtColor(testImg, cv2.COLOR_RGB2GRAY)
        cannyImg = cv2.Canny(grayImg, 100, 130)
        # cv2.imshow("cannyImg", cannyImg)

        detector = Detector()
        black01 = cv2.cvtColor(np.zeros((rows, cols), dtype=np.uint8), cv2.COLOR_GRAY2BGR)

        ##image:8位，单通道图像。如果使用彩色图像，需要先转换为灰度图像。
        # method：定义检测图像中圆的方法。目前唯一实现的方法是cv2.HOUGH_GRADIENT。
        # dp：累加器分辨率与图像分辨率的反比。dp获取越大，累加器数组越小。
        # minDist：检测到的圆的中心，（x,y）坐标之间的最小距离。如果minDist太小，则可能导致检测到多个相邻的圆。如果minDist太大，则可能导致很多圆检测不到。
        # param1：用于处理边缘检测的梯度值方法。
        # param2：cv2.HOUGH_GRADIENT方法的累加器阈值。阈值越小，检测到的圈子越多。
        # minRadius：半径的最小大小（以像素为单位）。
        # maxRadius：半径的最大大小（以像素为单位）。
        circles = cv2.HoughCircles(cannyImg, cv2.HOUGH_GRADIENT, 5, 10,
                                   param1=60, param2=5, minRadius=1, maxRadius=10)
        ###过滤出在预置圆内的圆
        resCircles = detector.filteCircles(circles, CENTER, RADIUS * 2)
        meanCenter = [0, 0]
        if len(resCircles) != 0:
            resCircles = np.array(resCircles)
            centerCircles = resCircles[:, :2]
            meanCenter = np.mean(centerCircles, axis=0)
            # cv2.circle(testImg, (int(meanCenter[0]), int(meanCenter[1])), 5, (255, 255, 0), 2)
        else:
            print("no detect circles...")

        ### 直线检测
        lsd = cv2.createLineSegmentDetector(0)
        lines = lsd.detect(cannyImg)[0]
        if lines is None:
            print("no detector line.....")
        else:
            lines1 = lines[:, 0, :]  # 提取为二维
            lenLines = detector.filteLineWithLength(lines1, RADIUS)
            ###根据预置圆过滤
            cirLines = detector.filteLineWithCircle(lenLines, CENTER, RADIUS * 5)
            # print("cirLines num: ", len(cirLines))
            angleLines = detector.fiteWithAngle(cirLines, rows, cols)
            print("angleLines num: ", len(angleLines))
            # LineTools.drawLines(black01, angleLines)
            # LineTools.drawLines(testImg, angleLines)

            resLines = detector.resLines(angleLines, meanCenter, RADIUS * 5)
            LineTools.drawLines(testImg, resLines, 3)

        # cv2.circle(testImg, CENTER, RADIUS * 3, (255, 255, 255), 1)
        # cv2.circle(black01, CENTER, RADIUS * 3, (255, 255, 255), 1)

        # cv2.imwrite(resPath + files[i], testImg)
        # cv2.imwrite(resPath + files[i], black01)
        # cv2.imshow("black01", black01)
        cv2.imshow('testImg', testImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    runMain()
    # myRename()
