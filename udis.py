from __future__ import print_function
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
import os

img_names_undistort = [img for img in glob.glob(
    "./*.png")]
new_path = "./undist/"

camera_matrix = np.array([[9.19377115e+03, 0.00000000e+00, 8.45716058e+02],
                          [0.00000000e+00, 9.86542071e+03, 5.66083591e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coefs = np.array([-1.73942236e+01,  6.06457359e+02, -2.43299457e-01,  3.36809040e-02,
                       -1.69849843e+04])

i = 0

# for img_found in img_names_undistort:
while i < len(img_names_undistort):
    img = cv2.imread(img_names_undistort[i])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coefs, (w, h), 1, (w, h))

    # dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

    # dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    # undistort
    mapx, mapy = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coefs, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    # crop and save the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]

    name = img_names_undistort[i].split("\\")
    name = name[-1]
    full_name = new_path + name + '_undist.png'

    # outfile = img_names_undistort + '_undistorte.png'
    print('Undistorted image written to: %s' % full_name)
    cv2.imwrite(full_name, dst)
    i = i + 1
