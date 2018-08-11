import sys
import cv2
import numpy as np


def stitching(img1, img2, M):

    # Узнаём ширину и высоту изображений
    width1, high1 = img1.shape[:2]
    width2, high2 = img2.shape[:2]

    # Вычисляем размер выходного изображения
    img1_temp_dim = np.float32([[0, 0], [0, width1], [high1, width1], [high1, 0]]).reshape(-1, 1, 2)
    img2_temp_dim = np.float32([[0, 0], [0, width2], [high2, width2], [high2, 0]]).reshape(-1, 1, 2)

    # Вычисляем отклонение камеры относительно первого изображения
    img2_dims = cv2.perspectiveTransform(img2_temp_dim, M)

    result_dims = np.concatenate((img1_temp_dim, img2_dims), axis=0)

    # Соединяем изображения
    [x_min, y_min] = np.int32(result_dims.min(axis=0).ravel() - 0.5)
    [x_max, y_max] = np.int32(result_dims.max(axis=0).ravel() + 0.5)

    transform_dist = [-x_min, -y_min]
    transform_array = np.array([[1, 0, transform_dist[0]],
                                [0, 1, transform_dist[1]],
                                [0, 0, 1]])

    result_img = cv2.warpPerspective(img2, transform_array.dot(M),
                                     (x_max - x_min, y_max - y_min))
    result_img[transform_dist[1]:width1 + transform_dist[1],
    transform_dist[0]:high1 + transform_dist[0]] = img1

    return result_img


def homography(img1, img2):
    # Используем алгоритм SIFT
    orb = cv2.ORB_create()

    # Вычисляем ключевые точки и дескрипторы
    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Выбираем совпадающие
    FLANN_INDEX_LSH = 6

    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    proper_matches = []
    for i, (m1, m2) in enumerate(matches):
        if m1.distance < 0.775 * m2.distance:
            proper_matches.append(m1)

    min_matches = 4
    if len(proper_matches) > min_matches:

        img1_pts = []
        img2_pts = []

        for match in proper_matches:
            img1_pts.append(keypoints1[match.queryIdx].pt)
            img2_pts.append(keypoints2[match.trainIdx].pt)
        img1_pts = np.float32(img1_pts).reshape(-1, 1, 2)
        img2_pts = np.float32(img2_pts).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(img1_pts, img2_pts, cv2.RANSAC, 5.0)
        return M
    else:
        return ()
        exit()


def stitcher(img1, img2):

    M = homography(img1, img2)
    result_image = stitching(img2, img1, M)
    return result_image
