# coding=utf-8
import cv2
import numpy as np
import os


def decode(patch):
    rows, cols = patch.shape[:2]
    thresh = round(cols/10)
    resLoc = []
    if rows*cols != 0:
        for i in range(2, cols - 2):
            if patch[0, i] == 255:
                temp01 = patch[:, (i - 1):(i + 1)]
                temp02 = patch[:, i:(i + 2)]
                if (rows == sum(temp01.ravel() == 255)) or (rows == sum(temp02.ravel() == 255)):
                    if (i < 8*thresh) and (i > 2*thresh):
                        resLoc.append(i)
    return resLoc


def firstFilter(imgCode, thresh):
    resCode = []
    tempCode = []
    for i in range(len(imgCode)):
        if i == len(imgCode)-1:
            if imgCode[i] - imgCode[i-1] >= thresh:
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

def secondFilter01(imageCode):
    resCode = []
    codeLen = len(imageCode)
    tempRes = []
    count = 0
    for i in range(codeLen):
        if i == codeLen -1:
            tempCode01 = imageCode[i - 1]
            tempCode02 = imageCode[i]

            len_tempCode02 = len(tempCode02)
            for j in range(1, len_tempCode02):
                tempRes.append(tempCode02[j])

            if tempCode02[0] - tempCode01[0] == 1:
                tempRes = list(set(tempRes))
                tempRes.insert(0, tempCode02[0] - count)
                tempRes.insert(1, count + 1)
                resCode.append(tempRes)
                # print("aa tempres: ", tempRes)
            else:
                tempRes = list(set(tempRes))
                tempRes.insert(0, tempCode02[0])
                tempRes.insert(1, 1)
                resCode.append(tempRes)
                # print("bb tempres: ", tempRes)
            break

        tempCode01 = imageCode[i]
        tempCode02 = imageCode[i+1]
        if tempCode02[0] - tempCode01[0] == 1:
            count = count + 1
            len_tempCode01 = len(tempCode01)
            for j in range(1, len_tempCode01):
                tempRes.append(tempCode01[j])
        else:
            len_tempCode01 = len(tempCode01)
            for j in range(1, len_tempCode01):
                tempRes.append(tempCode01[j])
            tempRes = list(set(tempRes))
            tempRes.insert(0, tempCode01[0]-count)
            tempRes.insert(1, count + 1)
            resCode.append(tempRes)
            # print("tempRes ", tempRes)
            tempRes = []
            count = 0

    print("secondFilter code ", resCode)
    return resCode

def secondFilter(imageCode, cols):
    tempLoc = []
    for tempCode in imageCode:
        for loc in tempCode[1:]:
            tempLoc.append(loc)
    tempLoc = np.array(tempLoc)
    tempLoc = tempLoc[tempLoc < (8 / 10 * cols)]
    tempLoc = tempLoc[tempLoc > (2 / 10 * cols)]
    tempLoc = tempLoc.tolist()
    tempLocSet = set(tempLoc)
    tempLoc.sort()
    print("ccc temploc ", tempLoc)


def thirdFilter(imageCode, cols):
    for codeTemp in imageCode:
        codeLoc = np.array(codeTemp[2:-1])
        codeLoc = codeLoc[codeLoc<(8/10*cols)]
        codeLoc = codeLoc[codeLoc > (2 / 10 * cols)]

def DecodePatchImage(image, bins, thresh01, thresh02):
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
        patchImgCode = decode(patchImg)
        # print("patch" + str(i), "code: ", patchImgCode)
        if len(patchImgCode) > 1:
            patchImgCode = firstFilter(patchImgCode, round(cols / 30))
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


def runMain():
    resPath = "F:/NARI/Data/sichuan/braker/firstDetImages/dstTemp/"
    dirPath = "F:/NARI/Data/sichuan/braker/firstDetImages/vertical/"
    files = os.listdir(dirPath)
    num = len(files)
    lsd = cv2.createLineSegmentDetector(0)
    for i in range(int(num)):
        print("#####第" + str(i + 1) + "张图像: ", files[i], "#####")
        [fileName, fileFormat] = files[i].split(".")
        testImg = cv2.imread(dirPath + files[i])
        # testImg = cv2.imread(dirPath + "firstBraker5.jpg")
        rows, cols = testImg.shape[:2]
        print("rows: ", rows, ", cols:", cols)

        blurred = cv2.GaussianBlur(testImg, (3, 3), 0)
        grayImg = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        testEdges = cv2.Canny(grayImg, 100, 200)
        cv2.imshow("testEdges", testEdges)
        firstCode = DecodePatchImage(testEdges, 40, 10, 0.9)
        print(firstCode)
        secondFilter(firstCode, cols)

        ## 画线
        if len(firstCode) != 0:
            height = round(rows / 40)
            for pCode in firstCode:
                # print("pcode: ", pCode)
                # for i in range(1, len(pCode)):
                for i in range(1,2):
                    pt1 = (pCode[i], pCode[0] * height)
                    pt2 = (pCode[i], (pCode[0] + 1) * height)
                    cv2.line(testImg, pt1, pt2, (0, 0, 255), 2)
        else:
            print("no detector braker.......")

        cv2.imshow("testImg", testImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    runMain()
