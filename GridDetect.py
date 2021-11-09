import math
import cv2
import imutils as imutils
import numpy as np
from alanTuning import *

def detect_grid(capture):
    centralPoint = [0, 0]
    squareRadius = 0
    angle = 0

    referenceFrame = 0
    referenceFrameWithBuffer = 0
    lastSquares = []
    stabilityCounter = 0

    newContours = []
    newContourCentres = []
    newContourRadians = []
    newContourLens = []
    newContourPoints = []

    for i in range(AT_calibrationFrames):

        ret, frame = capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
        binary2 = 255 - binary
        contours2 = cv2.findContours(binary2, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours + contours2:
            arcLen = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * arcLen, True)
            if len(approx) == 4 and 1080 / 8 * 4 / 2 < arcLen < 1080 / 8 * 4:
                centre = np.mean(approx, axis=0)
                goodCorners = 0
                for x in approx:
                    if 0.9 < np.sum(np.square(centre - x)) / (arcLen / 8 * 1.4142) ** 2 < 1.1:
                        goodCorners += 1
                if goodCorners == 4:
                    centreDistances = [np.sum(np.square(centre - x)) for x in newContourCentres]
                    if min(centreDistances + [200]) > 100:
                        newContours.append(approx)
                        newContourCentres.append(centre)
                        oneAngle = math.atan2((approx[0][0] - approx[1][0])[0], (approx[0][0] - approx[1][0])[1])
                        oneAngleDegrees = oneAngle * 180 / 3.14 % 90
                        if oneAngleDegrees > 45:
                            oneAngleDegrees = oneAngleDegrees - 90
                        newContourRadians.append(oneAngleDegrees)
                        newContourLens.append(arcLen / 4)
                        for i in approx:
                            for j in approx:
                                newContourPoints.append(i[0] * 2 - j[0])
        if len(newContours) > 0:
            for (x, y) in newContourPoints:
                frame[y - 2:y + 2, x - 2:x + 2] = (0, 0, 255)

            frameCentre = [960, 540]
            pointDistance = [np.sum(np.square(np.subtract(point, frameCentre))) for point in newContourPoints]
            centralPoint = newContourPoints[np.argmin(pointDistance)]
            squareRadius = int(np.median(newContourLens) * 4.5)

            frame[frameCentre[1] - 5:frameCentre[1] + 5, frameCentre[0] - 5:frameCentre[0] + 5] = (0, 255, 255)
            frame[centralPoint[1] - 5:centralPoint[1] + 5, centralPoint[0] - 5:centralPoint[0] + 5] = (0, 255, 255)


        frame = cv2.resize(frame, (960, 540))
        cv2.imshow("contours", frame)

        angle = np.median(newContourRadians)
        rotated = imutils.rotate(frame, -angle)

        cropped = rotated[centralPoint[1] // 2 - squareRadius // 2:centralPoint[1] // 2 + squareRadius // 2,
                  centralPoint[0] // 2 - squareRadius // 2:centralPoint[0] // 2 + squareRadius // 2]
        cv2.imshow("cropped", cropped)
        cv2.waitKey(1)

    cv2.destroyAllWindows()
    return centralPoint, squareRadius, angle


def rotate_crop_frame(frame, centralPoint, squareRadius, angle):
    rotated = imutils.rotate(frame, -angle)
    cropped = rotated[centralPoint[1] - squareRadius:centralPoint[1] + squareRadius, centralPoint[0] - squareRadius:centralPoint[0] + squareRadius]
    return cropped


def draw_grid(img_in, color=(0, 255, 0), thickness=1):
    img = np.array(img_in)
    h, w = img.shape
    rows, cols = 8, 8
    dy, dx = h / rows, w / cols
    for x in np.linspace(start=dx, stop=w - dx, num=cols - 1):
        x = int(round(x))
        cv2.line(img, (x, 0), (x, h), color=color, thickness=thickness)
    for y in np.linspace(start=dy, stop=h - dy, num=rows - 1):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)
    return img