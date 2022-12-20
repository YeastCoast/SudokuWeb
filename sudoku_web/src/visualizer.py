import cv2
import numpy as np
import operator
from matplotlib import pyplot as plt

def draw_digits(solved_array, solved_coords, transformed_grid):
    w, h, c = transformed_grid.shape
    base_w = w / 9
    base_h = h / 9
    image = transformed_grid
    for coord in solved_coords:
        i = coord[0]
        j = coord[1]
        p1 = (j * base_w, i * base_h)  # Top left corner of a bounding box
        p2 = ((j + 1) * base_w, (i + 1) * base_h)  # Bottom right corner of bounding box
        center = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
        text_size, _ = cv2.getTextSize(str(solved_array[i, j]), cv2.FONT_HERSHEY_SIMPLEX, 0.75, 5)
        text_origin = (int(center[0] - text_size[0] // 2), int(center[1] + text_size[1] // 2))
        image = cv2.putText(image, str(solved_array[i, j]),
                    text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    return image

def find_extreme_corners(polygon, limit_fn, compare_fn):
    # limit_fn is the min or max function
    # compare_fn is the np.add or np.subtract function

    # if we are trying to find bottom left corner, we know that it will have the smallest (x - y) value
    section, _ = limit_fn(enumerate([compare_fn(pt[0][0], pt[0][1]) for pt in polygon]),
                          key=operator.itemgetter(1))

    return polygon[section][0][0], polygon[section][0][1]

def draw_extreme_corners(pts, original):
    cv2.circle(original, pts, 7, (0, 255, 0), cv2.FILLED)

def find_contours(img, original):
    # find contours on thresholded image
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # sort by the largest
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    polygon = None

    # make sure this is the one we are looking for
    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, closed=True)
        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, closed=True)
        num_corners = len(approx)

        if num_corners == 4 and area > 1000:
            polygon = cnt
            break

    if polygon is not None:
        # find its extreme corners
        top_left = find_extreme_corners(polygon, min, np.add)  # has smallest (x + y) value
        top_right = find_extreme_corners(polygon, max, np.subtract)  # has largest (x - y) value
        bot_left = find_extreme_corners(polygon, min, np.subtract)  # has smallest (x - y) value
        bot_right = find_extreme_corners(polygon, max, np.add)  # has largest (x + y) value

        # if its not a square, we don't want it
        if bot_right[1] - top_right[1] == 0:
            return []
        if not (0.95 < ((top_right[0] - top_left[0]) / (bot_right[1] - top_right[1])) < 1.05):
            return []

        cv2.drawContours(original, [polygon], 0, (0, 0, 255), 3)

        # draw corresponding circles
        [draw_extreme_corners(x, original) for x in [top_left, top_right, bot_right, bot_left]]

        return [top_left, top_right, bot_right, bot_left]

    return []


def unwarp_image(img_src, img_dest, pts):
    pts = np.array(pts)

    height, width = img_src.shape[0], img_src.shape[1]
    pts_source = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, width - 1]],
                          dtype='float32')
    #print(pts, pts_source)
    h, status = cv2.findHomography(pts_source, pts)
    warped = cv2.warpPerspective(img_src, h, (img_dest.shape[1], img_dest.shape[0]))
    #plt.imshow(warped)
    #plt.show()
    cv2.fillConvexPoly(img_dest, np.int32(pts), 0, 16)

    dst_img = cv2.add(img_dest, warped)

    dst_img_height, dst_img_width = dst_img.shape[0], dst_img.shape[1]

    return dst_img