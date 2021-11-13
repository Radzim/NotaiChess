import math
import cv2.cv2 as cv2
import imutils as imutils
import numpy as np
from alanTuning import *

def detect_grid(capture):
    for i in range(AT_calibrationFrames):
        # INITIALISING
        newContourRadians = []
        newContourLens = []
        newContourPoints = []
        # READ FRAME
        ret, frame = capture.read()
        if not ret:
            return 'ERROR 1B: video stream stopped'
        height, width = np.shape(frame)[:2]
        print(width, height)
        # GET CONTOURS
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
        contours2 = cv2.findContours(255 - binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]
        # FILTER CONTOURS - BOARD ONLY
        for contour in contours + contours2:
            arcLen = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.01 * arcLen, True)
            print(len(approx), arcLen)
            # SQUARES OF APPROXIMATELY GOOD SIZE ONLY
            if len(approx) == 4 and min(width, height) / 8 < arcLen / 4 < min(width, height) / 8 * 2:
                centre = np.mean(approx, axis=0)
                goodCorners = 0
                # CHECK SQUARENESS
                for x in approx:
                    if 0.9 < np.sum(np.square(centre - x)) / (arcLen / 8 * 1.4142) ** 2 < 1.1:
                        goodCorners += 1
                # ADD TO LIST OF FILTERED CONTOURS IF IS GOOD SQUARE
                if goodCorners == 4:
                        # CALCULATE AND ADD ANGLES
                        oneAngleRadians = math.atan2((approx[0][0] - approx[1][0])[0], (approx[0][0] - approx[1][0])[1])
                        oneAngleDegrees = oneAngleRadians * 180 / 3.14 % 90
                        if oneAngleDegrees > 45:
                            oneAngleDegrees = oneAngleDegrees - 90
                        newContourRadians.append(oneAngleDegrees)
                        newContourLens.append(arcLen / 4)
                        # ADD EXTENDED MESH OF PREDICTED GRID POINTS
                        for i in approx:
                            for j in approx:
                                newContourPoints.append(i[0] * 2 - j[0])
        # RETURN IF ENOUGH GRID POINTS DETECTED
        if len(newContourPoints) > 20:
            frameCentre = [width//2, height//2]
            squareRadius = int(np.median(newContourLens) * 4.5)
            pointDistance = [np.sum(np.square(np.subtract(point, frameCentre))) for point in newContourPoints]
            centralPoint = newContourPoints[np.argmin(pointDistance)]
            angle = np.median(newContourRadians)
            return centralPoint, squareRadius, angle
    return ('ERROR 2: Failed to detect grid')

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