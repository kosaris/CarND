import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
from moviepy.editor import VideoFileClip
import moviepy as mve
import road_lines

#load camera calibration parameters
pkl_file = open('wide_dist_pickle.p', 'rb')
pkl_data = pickle.load(pkl_file)
mtx = pkl_data["mtx"]
dist = pkl_data["dist"]

#load perspective transform parameters
pkl_file = open('perspetive_pickle.p', 'rb')
pkl_data = pickle.load(pkl_file)
M = pkl_data["M"]
M_inv = pkl_data["M_inv"]

my_road_line_detector = road_lines.RoadLineDetector(mtx, dist, M , M_inv)

#image = cv2.imread("test_images/straight_lines2.jpg")
#b,g,r = cv2.split(image)       # get b,g,r
#image = cv2.merge([r,g,b])     # switch it to rgb
#result = my_road_line_detector.process_image(image)
#plt.figure()
#plt.imshow(result)
#plt.show()

output = 'out_project_video1.mp4'
clip1 = VideoFileClip("project_video.mp4")
out_clip = clip1.fl_image(my_road_line_detector.process_image)
out_clip.write_videofile(output, audio=False)
