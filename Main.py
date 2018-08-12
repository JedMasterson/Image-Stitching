from os import listdir
import imutils
import cv2
import numpy as np
from Stitching_file import stitcher

# Находим все файлы в папке с расширением .jpg
images = []
list_files = [x for x in listdir() if x.endswith(".jpg")]

# Считываем их
for file in list_files:
    images.append(cv2.imread(file))

images_matches = []

# Выбираем первое изображение как стартовое
img1 = images.pop(0)
# Подбираем в цикле наиболее подходящее изображение по количеству совпадающих дескрипотов
while 0 < len(images):
    img2 = -1
    for number in images:
        images_matches.clear()
        orb = cv2.ORB_create(nfeatures=10000)
        kp1, d1 = orb.detectAndCompute(img1, None)
        kp2, d2 = orb.detectAndCompute(number, None)

        FLANN_INDEX_LSH = 6

        index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(d1, d2, k=2)

        verified_matches = 0
        for i, (m1, m2) in enumerate(matches):
            if m1.distance < 0.8 * m2.distance:
                verified_matches += 1
        images_matches.append(verified_matches)
        img2 = images_matches.index(max(images_matches))
    if img2 != -1:
        img2 = images.pop(img2)
        img1 = stitcher(img1, img2)

    else:
        break

img1 = imutils.resize(img1, width=1000)
cv2.imshow('Result', img1)
cv2.imwrite(r"C:\Users\Пкеуу\Pictures\Test_photo\Nastya\Result.jpg", img1)
cv2.waitKey()
