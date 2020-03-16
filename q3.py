import cv2
import glob
from matplotlib import pyplot as plt


if __name__ == "__main__":
    query_pics = []
    database = []

    real_image = cv2.imread("00461.png", 0)

    for img in glob.glob("PicturesQ3/*.png"):
        query_pics.append(cv2.imread(img, 0))

    for img in glob.glob("Database/*.png"):
        database.append(cv2.imread(img, 0))

    orb = cv2.ORB_create()

    kp = []
    des = []
    matches_array = []
    matches_len = []

    kp1, des1 = orb.detectAndCompute(query_pics[1], None)

    for i in range(len(database)):
        kp2, des2 = orb.detectAndCompute(database[i], None)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)
        matches_len.append(matches[0].distance)
        matches_array.append(matches)

    # print(matches_array[0])

    # distance = matches_array[0].
    # for m in matches_array:


    index = matches_len.index(min(matches_len))
    image_found = database[index]

    arr = matches_array[index]

    img3 = cv2.drawMatches(image_found, kp2, query_pics[1], kp1, arr[:0], None, flags=2)
    plt.imshow(img3, cmap='gray'), plt.show()
