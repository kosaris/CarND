import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import glob
from vehicle_detector import VehicleDetector
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pickle
from sklearn.externals import joblib
from vehicle_detection_library import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import sys
# args = [float(x) for x in sys.argv[1:]]
# start_time = args[0]
# stop_time = args[1]

svc = joblib.load('svc_ycrcb_pickle.p')

pkl_file = open('scaler_pickle.p', 'rb')
pkl_data = pickle.load(pkl_file)
X_scaler = pkl_data["X_scaler"]
# from skimage.feature import hog
color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
orient = 9  # HOG orientations
pix_per_cell = 8 # HOG pixels per cell
cell_per_block = 2 # HOG cells per block
hog_channel = "ALL" # Can be 0, 1, 2, or "ALL"
spatial_size = (16, 16) # Spatial binning dimensions
hist_bins = 16    # Number of histogram bins
spatial_feat = True # Spatial features on or off
hist_feat = True # Histogram features on or off
hog_feat = True # HOG features on or off
y_start_stop = [400, 496] # Min and max in y to search in slide_window()
image_size = (720, 1280)

my_vehicle_detector = VehicleDetector(image_size, svc, X_scaler, color_space,
                      spatial_size, hist_bins, orient, pix_per_cell, 
                      cell_per_block, hog_channel, spatial_feat, hist_feat, 
                      hog_feat)
  
output = "out_project_video.mp4"
clip1 = VideoFileClip("project_video.mp4").subclip(41, 41.2)
# clip1 = VideoFileClip("project_video.mp4")
out_clip = clip1.fl_image(my_vehicle_detector.process_image)
out_clip.write_videofile(output, audio=False)