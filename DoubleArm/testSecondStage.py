# coding=utf-8
import cv2
import os
import sys

# cwdPath = os.getcwd()
# print(cwdPath)
sys.path.append('./')
# print(sys.path)
from brakerArmDetector import detector


def drawResImg(image, resLoc):
    length = len(resLoc)
    for i in range(length):
        cv2.line(image, resLoc[i][0], resLoc[i][1], (255, 0, 0), 2)


def runMain():
    resPath = "F:/NARI/dlWork/houghDetector/hough/"
    dirPath = "F:/NARI/dlWork/houghDetector/FirstDetectionImages/"
    files = os.listdir(dirPath)
    num = len(files)
    for i in range(int(num)):
        print("#####第" + str(i + 1) + "张图像: ", files[i], "#####")
        [fileName, fileFormat] = files[i].split(".")
        imgOrigin = cv2.imread(dirPath + files[i])
        # imgOrigin = cv2.imread(dirPath + "Breaker_Arm_8.jpg")
        redDetector = detector(imgOrigin)
        if len(redDetector) != 0:
            drawResImg(imgOrigin, redDetector[1])
            cv2.imwrite(resPath + fileName + ".jpg", imgOrigin)
        else:
            print("detector result is null, draw Image failed!")
            # cv2.imshow("result", imgOrigin)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

# if __name__ == '__main__':
#     runMain()
