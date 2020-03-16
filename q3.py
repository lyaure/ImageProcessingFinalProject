import cv2
import glob
from matplotlib import pyplot as plt


if __name__ == "__main__":
    query_pics = []
    database = []
    final = []

    for img in glob.glob("PicturesQ3/*.png"):
        query_pics.append(cv2.imread(img, 0))

    for img in glob.glob("Database/*.png"):
        database.append(cv2.imread(img, 0))

    orb = cv2.ORB_create()

    kp = []
    des = []
    matches_array = []
    matches_len = []

    for i in range(len(query_pics)):
        kp1, des1 = orb.detectAndCompute(query_pics[i], None)

        for j in range(len(database)):
            kp2, des2 = orb.detectAndCompute(database[j], None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            matches_len.append(matches[0].distance)
            matches_array.append(matches)

        index = matches_len.index(min(matches_len))
        image_found = database[index]

        arr = matches_array[index]

        img3 = cv2.drawMatches(image_found, kp2, query_pics[i], kp1, arr[:0], None, flags=2)
        final.append(img3)

    for img in final:
        plt.imshow(img, cmap='gray')
        plt.show()
