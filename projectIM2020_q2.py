import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt


# The function finds and draws circles using hough circles algorithm
def hough_circles(image):
    cimg = image.copy()

    circles = cv2.HoughCircles(cimg, cv2.HOUGH_GRADIENT, 1, 20, param1=20, param2=55, minRadius=15, maxRadius=49)

    # draw circles if found
    if circles is not None:
        circles = np.uint16(np.around(circles))
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)

    return cimg


if __name__ == "__main__":
    images = []
    final = []

    # read images
    for img in glob.glob("PicturesQ2/*.png"):
        images.append(cv2.imread(img, 0))

    for img in images:
        final.append(hough_circles(img))

    # show images
    for i in range(len(images)):
        plt.subplot(121)
        plt.imshow(images[i], cmap='gray')
        plt.subplot(122)
        plt.imshow(final[i], cmap='gray')
        plt.show()
