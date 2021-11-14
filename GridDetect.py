import math
import cv2.cv2 as cv2
import imutils as imutils
import numpy as np
from alanTuning import *


def detect_grid(capture):
    filtered_pairs = []
    for i in range(AT_calibrationFrames):
        # READ FRAME
        ret, frame = capture.read()
        if not ret:
            return 'ERROR 1B: video stream stopped'
        height, width = np.shape(frame)[:2]
        # DETECT CORNER POINTS
        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kernel = np.array([[-1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, -1]]) / 4
        corner_filtered = cv2.filter2D(frame_grayscale, -1, kernel)
        corner_filtered_threshold = np.where(corner_filtered > np.max(corner_filtered) * 0.8, corner_filtered * 0 + 1, 0)
        # GET GRID
        all_points = np.argwhere(corner_filtered_threshold == 1)
        clustered_points = cluster_points(all_points, 2, min(height, width) / 100)
        extended_points = cluster_points(extend_points(clustered_points, height, width), 1, min(height, width) / 20)
        # ADD CORRECT PAIRS TO LIST
        extended_pairs = point_pairs(extended_points)
        pairwise_distances = [np.sum(np.square(exp[0]-exp[1]))**0.5 for exp in extended_pairs]
        for pd in range(len(pairwise_distances)):
            if min(width, height) > pairwise_distances[pd]*4*1.1412 > min(width, height)/2:
                filtered_pairs.append(extended_pairs[pd])
    # EDGE CASE WHEN NO OR FEW GRIDS DETECTED
    if len(filtered_pairs) < 5:
        return ('ERROR 2: Failed to detect grid')
    # FIND GRID SIZE
    filtered_distances = [np.sum(np.square(exp[0]-exp[1]))**0.5 for exp in filtered_pairs]
    filtered_points = np.array(filtered_pairs).reshape(-1, 2)
    distance = int(np.median(filtered_distances)/1.4142*4.5)
    # FIND GRID ANGLES
    pairwise_angles = [math.atan2((pair[0]-pair[1])[0], (pair[0]-pair[1])[1]) * 180 / 3.14 % 90 for pair in filtered_pairs]
    angle = np.median(pairwise_angles)-45
    # FIND CENTRAL GRID POINT
    reference_centre = np.array(np.mean(filtered_points, axis=0), dtype=int) # reference_centre = np.array([height//2, width//2])
    centre = filtered_points[np.argmin([np.sum(np.square(exp - reference_centre))**0.5 for exp in filtered_points])]
    return centre, distance, angle

def rotate_crop_frame(frame, centralPoint, squareRadius, angle):
    border = cv2.copyMakeBorder(
        frame,
        top=int(squareRadius*1.5),
        bottom=int(squareRadius*1.5),
        left=int(squareRadius*1.5),
        right=int(squareRadius*1.5),
        borderType=cv2.BORDER_CONSTANT,
        value=[0, 0, 0]
    )
    centred = border[centralPoint[0]:centralPoint[0] + squareRadius*3,
              centralPoint[1]:centralPoint[1] + squareRadius*3]
    rotated = imutils.rotate(centred, angle)
    cropped = rotated[int(squareRadius*0.5):int(squareRadius*2.5),
              int(squareRadius*0.5):int(squareRadius*2.5)]
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


def cluster_points(all_points, minimum_points, max_distance):
    z = np.array([complex(c[0], c[1]) for c in all_points])
    m, n = np.meshgrid(z, z)
    dists = abs(m - n)
    dists_cluster = np.where(dists <= max_distance, 1, 0)
    all_points_xs, all_points_ys = [ap[0] for ap in all_points], [ap[1] for ap in all_points]
    new_point_xs = [sum(np.where(one_line_cluser, all_points_xs, 0)) // sum(one_line_cluser) for one_line_cluser in
                    dists_cluster]
    new_point_ys = [sum(np.where(one_line_cluser, all_points_ys, 0)) // sum(one_line_cluser) for one_line_cluser in
                    dists_cluster]
    new_points = list(zip(new_point_xs, new_point_ys))
    point_set = set()
    for n_point in new_points:
        if new_points.count(n_point) >= minimum_points:
            point_set.add(n_point)
    return np.array(list(point_set))


def extend_points(all_points, height, width):
    ret_points = []
    for a in all_points:
        for b in all_points:
            if (a - b)[0] ** 2 + (a - b)[1] ** 2 < min(height / 5, width / 5) ** 2:
                if 0 < (a + a - b)[1] < width and 0 < (a + a - b)[0] < height:
                    ret_points.append(a + a - b)
                if 0 < (b + b - a)[1] < width and 0 < (b + b - a)[0] < height:
                    ret_points.append(b + b - a)
    return ret_points


def point_pairs(extended_points):
    ret_pairs = []
    z = np.array([complex(c[0], c[1]) for c in extended_points])
    m, n = np.meshgrid(z, z)
    dists = np.where(abs(m - n) == 0, 100000, abs(m - n))
    for i in range(len(dists)):
        ret_pairs.append((extended_points[i], extended_points[dists[i].argmin()]))
    return ret_pairs