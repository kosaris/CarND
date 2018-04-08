import pickle
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont

def hls_select(img, thresh=(0, 255)):
    """ selects the pixels of img that are in the treshold
    
    Args:
    img : input image
    thresh: (lower, upper) values between the lower and upper threshold are selected
    
    Returns:
    An image with values inside the threshold set to 1 and the rest set to 0
    """
    # 1) Convert to HLS color space
    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    S = hls[:, :, 2]
    binary = np.zeros_like(S)
    binary[(S > thresh[0]) & (S <= thresh[1])] = 1
    return binary

def sobel(img, thresh=(50,100)):
    """ Applies sobel transform in the x direction
    
    operates on the image, applies sobel transform in the x direction, takes the 
    absolute value, scales the result and sets the values between thresh[0] and
    thresh[1] to 1 and the rest to zero 
    
    Args:
    img : input image
    thresh: (lower, upper) values between the lower and upper threshold are selected
    
    Returns:
    An image with values inside the threshold set to 1 and the rest set to 0
    """
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    thresh_min = thresh[0]
    thresh_max = thresh[1]
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1
    return sxbinary
        
class RoadLineDetector():
    """ class for detecting road lines.
     
     Provides tools for detecting the line segments, using histogram method
     which does not use the previously found lines and a method that relies on
     the previously found lines.
    """
    def __init__(self, mtx, dist, M, M_inv):
        # mts distrotion paramter
        self.mtx = mtx
        # dist distortion parameter
        self.dist = dist
        # perspective transform
        self.M = M
        # inverse perspective transform
        self.M_inv = M_inv
        # flag to indicate successful detection of lines in the previous frame
        self.prev_detection = False
        # margin used in searching around the previous lines points in the new
        # frame
        self.margin = 100
        # coefficients of the left line
        self.left_fit = np.array([4.59384120e-05 , -2.06714118e-01 ,  4.13968001e+02])
        # coefficients of the right line
        self.right_fit = np.array([1.17955827e-04 , -2.79696905e-01 ,  1.25397792e+03])
        # gain of the filter for line coefficients. 
        # out = filter_gain * previous + (1-filter_gain)*current
        self.filter_gain = 0.8*0
        # number of points that must be detected to constitute a successful
        # detection
        self.min_index = 1000
    
    def get_combined_binary(self, road_img_undistort):
        """ combines the hls selector and sobel transform to create one image
        for better tracking of the lines
        
        Args:
        road_img_undistort: image of the road after distortion correction
        
        Returns:
        image of the road that combines the result of sobel and hls selection 
        and has been transformed using the perspective transform
        """
        
        hls_binary = hls_select(road_img_undistort, thresh=(120, 255))
        sxbinary = sobel(road_img_undistort, thresh=(50,100))
        # Stack each channel to view their individual contributions in green and blue respectively
        # This returns a stack of the two binary images, whose components you can see as different colors
        color_binary = np.dstack(( np.zeros_like(sxbinary), hls_binary , sxbinary )) * 255
        
        # Combine the two binary thresholds
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(hls_binary == 1) | (sxbinary == 1)] = 1
        
        img_size = (road_img_undistort.shape[1], road_img_undistort.shape[0])

        warped = cv2.warpPerspective(combined_binary, self.M, img_size)
        return warped
    
    def get_fresh_line_inds(self, warped):
        """ determines the points corresponding to the left and right line
        using the histogram of the image
        
        Uses the histogram of the image to determine the position of x of the
        two lines and uses this position with moving windows to track the line
        on the image and determine the points that correspond to the left and 
        right lines. It does not require the history of the line.  
        
        Args:
        warped: image of the road after applying distortion and perspective
        correction
        
        Returns:
        left_lane_inds: index of the points corresponding to the left line
        right_lane_inds: index of the points corresponding to the right line
        nonzerox: array of x for nonzero points
        nonzeroy: array of y for nonzero points
        """
        histogram = np.sum(warped[warped.shape[0]//2:,:], axis=0)
    
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        
        # Choose the number of sliding windows
        nwindows = 12
        # Set height of windows
        window_height = np.int(warped.shape[0]/nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base

        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        # Step through the windows one by one and find the nonzero points in each box
        # Append the points in the list and use them to fit a line to all the points
        # that fit in either of the boxes. 
        # Use the center of the nonzero points in each box to determine the margins for
        # the next window (x) 
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = warped.shape[0] - (window+1)*window_height
            win_y_high = warped.shape[0] - window*window_height
            win_xleft_low = leftx_current - self.margin
            win_xleft_high = leftx_current + self.margin
            win_xright_low = rightx_current - self.margin
            win_xright_high = rightx_current + self.margin

            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
            (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            
        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        if ((len(left_lane_inds)> self.min_index) & (len(right_lane_inds)> self.min_index)):
            self.prev_detection = True
        return left_lane_inds, right_lane_inds, nonzerox, nonzeroy
            
    def get_line_inds_using_old(self, warped):
        """ determines the points corresponding to the left and right line
        using the data from the previous sample
        
        this function uses the previous curve fit data to determine the points
        corresponding to the lines in the new frame. It should only be used when
        a the lines have been successfully detected in a previous frame 
        otherwise get_fresh_line_inds should be used.
        
        Args:
        warped: image of the road after applying distortion and perspective
        correction
        
        Returns:
        left_lane_inds: index of the points corresponding to the left line
        right_lane_inds: index of the points corresponding to the right line
        nonzerox: array of x for nonzero points
        nonzeroy: array of y for nonzero points
        """
        nonzero = warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []
        
        left_lane_inds = ((nonzerox > (self.left_fit[0]*(nonzeroy**2) + self.left_fit[1]*nonzeroy + 
        self.left_fit[2] - self.margin)) & (nonzerox < (self.left_fit[0]*(nonzeroy**2) + 
        self.left_fit[1]*nonzeroy + self.left_fit[2] + self.margin))) 
        
        right_lane_inds = ((nonzerox > (self.right_fit[0]*(nonzeroy**2) + self.right_fit[1]*nonzeroy + 
        self.right_fit[2] - self.margin)) & (nonzerox < (self.right_fit[0]*(nonzeroy**2) + 
        self.right_fit[1]*nonzeroy + self.right_fit[2] + self.margin)))  
        
        # Concatenate the arrays of indices
        if ((len(left_lane_inds)< self.min_index) | (len(right_lane_inds)< self.min_index)):     
            self.prev_detection = False
            return
        else:
            self.prev_detection = True

        return left_lane_inds, right_lane_inds, nonzerox, nonzeroy
    
    def process_image(self, road_img):
        """ processes the image to detect the lines and determine the curvature
        
        Args:
        road_img: the image of the road
        
        Returns:
        an image with the area between the two lines detected and overlayed on
        the original image, as well as the curvature of the two lines
        """
        road_img_undistort = cv2.undistort(road_img, self.mtx, self.dist, None, self.mtx)
        img_size = (road_img_undistort.shape[1], road_img_undistort.shape[0])

        warped = self.get_combined_binary(road_img_undistort)
        
        if (self.prev_detection):
            left_lane_inds , right_lane_inds, nonzerox, nonzeroy = self.get_line_inds_using_old(warped)
        if (not self.prev_detection):
            left_lane_inds , right_lane_inds, nonzerox, nonzeroy = self.get_fresh_line_inds(warped)
        
        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        
        # Fit a second order polynomial to each
        self.left_fit = self.filter_gain * self.left_fit + (1-self.filter_gain)* np.polyfit(lefty, leftx, 2)
        self.right_fit = self.filter_gain * self.right_fit + (1-self.filter_gain) * np.polyfit(righty, rightx, 2)
        
        # Generate x and y values for plotting
        ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0] )
        left_fitx = self.left_fit[0]*ploty**2 + self.left_fit[1]*ploty + self.left_fit[2]
        right_fitx = self.right_fit[0]*ploty**2 + self.right_fit[1]*ploty + self.right_fit[2]
        
        # Define conversions in x and y from pixels space to meters
        ym_per_pix = 30/720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meters per pixel in x dimension
        
        left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix, 2)
        y_eval = np.max(ploty)
        
        # Calculate the radius
        left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        curvature_msg = "left curve {:4f}, right curve {:4f}".format(left_curverad, right_curverad)
        
        color_warped = cv2.warpPerspective(road_img_undistort, self.M, img_size)
        
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))
        cv2.fillPoly(color_warped, np.int_([pts]), (255,0, 0))
        new_warp = cv2.warpPerspective(color_warped, self.M_inv, (color_warped.shape[1], color_warped.shape[0])) 
        result = cv2.addWeighted(road_img_undistort, 1, new_warp, 0.3, 0)
        texted_image =cv2.putText(img=np.copy(result), text=curvature_msg, org=(10,10),fontFace=1, fontScale=1, color=(255,255,255), thickness=2)
        return texted_image
