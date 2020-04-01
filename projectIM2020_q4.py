import numpy as np
import cv2
from matplotlib import pyplot as plt


# The function finds using hough circles algorithm
def hough_circles(original, maxRadius):
    image = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
    # hough circles
    circles = cv2.HoughCircles(original, cv2.HOUGH_GRADIENT, 1, 15, param1=20, param2=55, minRadius=15, maxRadius=maxRadius)
    draw_circle(circles, image)

    return image


# draw circles
def draw_circle(circles, image):
    if circles is not None:
        circles = np.uint16(np.around(circles))

        for circle in circles[0, :]:
            # draw the outer circle
            cv2.circle(image, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(image, (circle[0], circle[1]), 2, (0, 0, 255), 3)


if __name__ == "__main__":
    images = []
    titles = []

    images.append(cv2.imread("PicturesQ4/00223.jpg", 0))
    titles.append("Original")

    images.append(hough_circles(images[0], 35))
    titles.append("MaxRadius = 35")

    images.append(hough_circles(images[0], 50))
    titles.append("MaxRadius = 50")

    for i in range(len(images)):
        plt.subplot(1, len(images), i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])

    plt.show()
