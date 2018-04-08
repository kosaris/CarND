import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob

#load camera calibration parameters
pkl_file = open('wide_dist_pickle.p', 'rb')
pkl_data = pickle.load(pkl_file)
mtx = pkl_data["mtx"]
dist = pkl_data["dist"]

#read the image and apply distortion correction and convert to grayscale
road_img = mpimg.imread('test_images/test2.jpg')
road_img_undistort = cv2.undistort(road_img, mtx, dist, None, mtx)

#identify the corners of a rectangle from one image
corners= [[578,461],[710,467],[214,717],[1097,719]]
src = np.float32(corners)
img_size = (road_img_undistort.shape[1], road_img_undistort.shape[0])
offset_x = 250
offset_y = 20

dst = np.float32([[offset_x, offset_y], [img_size[0]-offset_x, offset_y], 
                             [offset_x, img_size[1]-offset_y],
                             [img_size[0]-offset_x, img_size[1]-offset_y]])

#get the perspective transform for these images
M = cv2.getPerspectiveTransform(src, dst)
M_inv = cv2.getPerspectiveTransform(dst, src)

#store the results
perspective_pickle = {}
perspective_pickle["src"] = src
perspective_pickle["dst"] = dst
perspective_pickle["M"] = M
perspective_pickle["M_inv"] = M_inv
pickle.dump( perspective_pickle, open( "perspetive_pickle.p", "wb" ) )