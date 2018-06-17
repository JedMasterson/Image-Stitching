from os import listdir
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
        orb = cv2.ORB_create()
        kp1, d1 = orb.detectAndCompute(img1, None)
        kp2, d2 = orb.detectAndCompute(number, None)

        bf = cv2.BFMatcher()
        k = 2
        matches = bf.knnMatch(d1, d2, k)

        verified_matches = 0
        for m1, m2 in matches:

            if m1.distance < 0.8 * m2.distance:
                verified_matches += 1
        images_matches.append(verified_matches)
        img2 = images_matches.index(max(images_matches))
    if img2 != -1:
        img2 = images.pop(img2)
        img1 = stitcher(img1, img2)
    else:
        break

cv2.imshow('Result', img1)
cv2.imwrite('Result.jpg', img1)
cv2.waitKey()
