from __future__ import print_function
import numpy as np
import cv2
import glob

img_names_undistort = [img for img in glob.glob(
    "./img_test/*.png")]
new_path = "./undist/"

camera_matrix = np.array([[484.70471536,   0.        , 538.38023713],
       [  0.        , 482.98833345, 544.09372506],
       [  0.        ,   0.        ,   1.        ]])
dist_coefs = np. array([ 0.31932615, -0.0646646 , -0.00164196, -0.00349879, -0.04961284])


for i, img_path in enumerate(img_names_undistort):
    img = cv2.imread(img_path)

    h,  w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coefs, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, camera_matrix, dist_coefs, None, newcameramtx)

    # get filename
    name = '.'.join(img_path.split("\\")[-1].split('.')[:-1])

    undst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)

    # crop and save the image
    x, y, w, h = roi
    cv2.imwrite(new_path + name + '_undist_0.png', dst)
    dst = dst[y:y+h, x:x+w]
    cv2.imwrite(new_path + name + '_undist_1.png', dst)

    # outfile = img_names_undistort + '_undistorte.png'
    print(f'Undistorted image: {img_path},\tProgress: {round((i+1)/len(img_names_undistort)*100,2)}%')
