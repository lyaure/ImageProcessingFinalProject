import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt


def histogram_equalization(image):
    img = image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    # plt.plot(cdf_normalized, color='b')
    # plt.hist(img.flatten(), 256, [0, 256], color='r')
    # plt.xlim([0, 256])
    # plt.legend(('cdf', 'histogram'), loc='upper left')
    # plt.show()

    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')

    return cdf[image]
    # plt.imshow(img2, cmap='gray')
    # plt.show()


def clean(image):
    bw_image = image.copy()
    bw_image = cv2.cvtColor(bw_image, cv2.COLOR_RGB2GRAY)
    # bw_image = histogram_equalization(bw_image)

    # print(cv2.mean(bw_image)[0])

    if cv2.mean(bw_image)[0] > 127:
        bw_image = cv2.bitwise_not(bw_image)

    # ret, th = cv2.threshold(images[1], 30, 255, cv2.THRESH_BINARY)

    closing = cv2.morphologyEx(bw_image, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))

    # ######################### effacer ####################
    # plt.imshow(closing, cmap='gray')
    # plt.show()
    # ######################### effacer ####################

    return closing


def find_corners(cleaned, image):
    ret = image.copy()

    img = cv2.bilateralFilter(cleaned, 3, 17, 17)
    edged = cv2.Canny(img, 20, 100)

    _, contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)

        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            if 4000 < area < 10000:
                for corner in approx:
                    cv2.circle(ret, (corner[0][0], corner[0][1]), 7, (255, 0, 0), 5)

    return ret


def find_lines(image):
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)

    h_lines = []
    v_lines = []

    angles = []

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

            angle = np.arctan2(y2 - y1, x2 - x1)*180. / np.pi
            angles.append(angle)

            if 0 <= angle <= 1 or -1 <= angle <= 0:
                h_lines.append((y1+y2)/2)
            elif 89 <= angle <= 91 or -91 < angle < -89:
                v_lines.append((x1+x2)/2)

    h_lines.sort()
    v_lines.sort()

    min_h = min(h_lines)
    max_h = max(h_lines)
    min_v = min(v_lines)
    max_v = max(v_lines)

    return min_h, max_h, min_v, max_v


def crop(image, min_h, max_h, min_v, max_v):
    height = image.shape[0]
    width = image.shape[1]

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


def add_black_pixels(image, min_h, max_h, min_v, max_v):
    new_img = image.copy()

    height = image.shape[0]
    width = image.shape[1]

    new_img[0:min_h, 0:min_v] = 0
    new_img[0:min_h, max_v:width] = 0
    new_img[max_h:height, 0:min_v] = 0
    new_img[max_h:height, max_v:width] = 0

    return new_img


if __name__ == '__main__':
    originals = []
    final = []
    titles = []

    for image in glob.glob("PicturesQ1/*"):
        originals.append(cv2.imread(image))

    for i in range(len(originals)):
        final.append(cv2.cvtColor(originals[i], cv2.COLOR_RGB2BGR))
        titles.append("Original")
        cleaned = clean(originals[i])

        ret,th = cv2.threshold(cleaned, 37, 255, cv2.THRESH_BINARY_INV)
        min_y, max_y, min_x, max_x = find_lines(th)

        new_img = histogram_equalization(final[i + 2*i])
        new_img = add_black_pixels(cleaned, int(min_y), int(max_y), int(min_x), int(max_x))
        corners = find_corners(new_img, final[i + 2*i])
        final.append(corners)
        titles.append("Chessboard")

        croped = crop(final[i + 2*i], int(min_y), int(max_y), int(min_x), int(max_x))
        final.append(croped)
        titles.append("Croped")

    for i in range(len(final)):
        plt.imshow(final[i])
        plt.title(titles[i])
        plt.show()


