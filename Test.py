from os import listdir
import numpy as np
import cv2
from matplotlib import pyplot as plt

list_files = [x for x in listdir(r"C:\Users\Пкеуу\Documents\GitHub\Image-Stitching\input_images") if x.endswith(".jpg")]
print(list_files[0])
cur_im = r"C:\Users\Пкеуу\Documents\GitHub\Image-Stitching\input_images\1.jpg"
img = cv2.imread(cur_im)
cv2.imshow("", img)
