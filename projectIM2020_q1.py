import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt


# The function makes a copy of the original image, turns it to gray colors and applies closing
def prep_image(image):
    bw_image = image.copy()
    bw_image = cv2.cvtColor(bw_image, cv2.COLOR_RGB2GRAY)

    # check if this is bright image
    if cv2.mean(bw_image)[0] > 127:
        # inverse colors
        bw_image = cv2.bitwise_not(bw_image)

    closing = cv2.morphologyEx(bw_image, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    return closing


# The function finds the corners of the chessboard and draws them
def find_corners(cleaned, image):
    final = image.copy()

    img = cv2.bilateralFilter(cleaned, 3, 17, 17)
    edged = cv2.Canny(img, 20, 100)

    # finds contours of shapes
    _, contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:  # checks only 4 sides shapes
            if 4000 < area < 10000: # selects only the chessboard rectangles
                for corner in approx:
                    # draws circles on the corners
                    cv2.circle(final, (corner[0][0], corner[0][1]), 7, (255, 0, 0), 5)

    return final


# The function finds all lignes in the image and selects only those from the ruler
def find_lines(image):
    # finds egdes
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    # finds lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    h_lines = []
    v_lines = []

    for i in range(len(lines)):
        for rho, theta in lines[i]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            # calculates angles of lines
            angle = np.arctan2(y2 - y1, x2 - x1)*180. / np.pi

            if 0 <= angle <= 1 or -1 <= angle <= 0: # horizonral lines
                h_lines.append((y1+y2)/2)
            elif 89 <= angle <= 91 or -91 < angle < -89: # vertical lines
                v_lines.append((x1+x2)/2)

    # select the first and last lines of the horizontal part of the ruler
    min_h = min(h_lines)
    max_h = max(h_lines)

    # select the first and last lines of the vertical part of the ruler
    min_v = min(v_lines)
    max_v = max(v_lines)

    return min_h, max_h, min_v, max_v


# The function crops the image according to points
def crop(image, min_h, max_h, min_v, max_v):
    height = image.shape[0]
    width = image.shape[1]

    # checks distances to keep the bigger part of image
    dst = height - max_h
    if dst > min_h:
        y1 = max_h
        y2 = height
    else:
        y1 = 0
        y2 = min_h

    dst = width - max_v
    if dst > min_v:
        x1 = max_v
        x2 = width
    else:
        x1 = 0
        x2 = min_v

    return image[y1:y2, x1:x2]


if __name__ == '__main__':
    originals = []
    final = []
    titles = []

    for image in glob.glob("PicturesQ1/*"):
        originals.append(cv2.imread(image))

    for i in range(len(originals)):
        final.append(cv2.cvtColor(originals[i], cv2.COLOR_RGB2BGR))
        titles.append("Original")
        cleaned = prep_image(originals[i])

        ret,th = cv2.threshold(cleaned, 37, 255, cv2.THRESH_BINARY)
        min_y, max_y, min_x, max_x = find_lines(th)

        corners = find_corners(cleaned, final[i + 2*i])
        final.append(corners)
        titles.append("Chessboard")

        croped = crop(final[i + 2*i], int(min_y), int(max_y), int(min_x), int(max_x))
        final.append(croped)
        titles.append("Croped")

    for i in range(len(final)):
        plt.imshow(final[i])
        plt.title(titles[i])
        plt.show()


