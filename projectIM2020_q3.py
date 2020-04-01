import cv2
import glob
from matplotlib import pyplot as plt


if __name__ == "__main__":
    query_pics = []
    database = []
    final = []

    # read images
    for img in glob.glob("PicturesQ3/*.png"):
        query_pics.append(cv2.imread(img, 0))

    for img in glob.glob("Database/*.png"):
        database.append(cv2.imread(img, 0))

    # initiate sift detector
    sift = cv2.xfeatures2d.SIFT_create()

    for i in range(len(query_pics)):
        matches_array = []
        matches_len = []

        # find the keypoints and descriptors with sift
        kp1, des1 = sift.detectAndCompute(query_pics[i], None)

        for j in range(len(database)):
            # find the keypoints and descriptors with sift
            kp2, des2 = sift.detectAndCompute(database[j], None)
            # BFMatcher
            bf = cv2.BFMatcher()
            matches = bf.match(des1, des2)
            # sort the matches according to distances
            matches = sorted(matches, key=lambda x: x.distance)
            # save the minimal distance
            matches_len.append(matches[0].distance)
            matches_array.append(matches)

        # find the index of the minimal distance of all images
        index = matches_len.index(min(matches_len))
        image_found = database[index]

        arr = matches_array[index]

        img3 = cv2.drawMatches(image_found, kp2, query_pics[i], kp1, arr[:0], None, flags=2)
        final.append(img3)

    for img in final:
        plt.imshow(img, cmap='gray')
        plt.show()
