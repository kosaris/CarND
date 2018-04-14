import numpy as np
from vehicle_detection_library import *
from scipy.ndimage.measurements import label

class VehicleDetector():
    def __init__(self, image_size, svc, X_scaler, color_space,
                      spatial_size, hist_bins, orient, pix_per_cell, 
                      cell_per_block, hog_channel, spatial_feat, hist_feat, 
                      hog_feat):
        self.image_size = image_size
        self.svc = svc
        self.X_scaler = X_scaler
        self.color_space = color_space
        self.spatial_size= spatial_size
        self.hist_bins = hist_bins
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block 
        self.hog_channel = hog_channel
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat
        
    def process_image(self, img):
        heat_map = np.zeros(self.image_size)
        hot_windows = []
        xy_windows = [(64,64), (64, 64), (96,96), (96,96), (128,128), (128,128)
                      , (196,196), (196,196)]
        
        y_start_stops = [[400,464], [416, 480], [400, 496], [432, 528],
                         [400, 528], [432, 560], [400, 596], [464, 660]]
        
        for xy_window, y_start_stop in zip(xy_windows, y_start_stops):
            hot_windows.append(get_hot_windows(img, y_start_stop, self.svc, 
                                          self.X_scaler, self.color_space,
                                          self.spatial_size, self.hist_bins, 
                                          self.orient, self.pix_per_cell, 
                                          self.cell_per_block, self.hog_channel, 
                                          self.spatial_feat, self.hist_feat, 
                                          self.hog_feat, xy_window))
            

        
        hot_windows = [item for sublist in hot_windows for item in sublist]
        
        heat_map = add_heat(heat_map, hot_windows)
        heat_mean = apply_threshold(heat_map, 2)
        heat_mean = np.clip(heat_mean, 0, 255)
        # Find final boxes from heatmap using label function
        labels = label(heat_mean)
        draw_img = draw_labeled_bboxes(np.copy(img), labels)

        return draw_img