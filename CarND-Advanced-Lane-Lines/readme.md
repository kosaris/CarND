**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/calibration1.png "camera calibration"
[image2]: ./output_images/distortion_correction_road1.png "Road Transformed"
[image3]: ./output_images/combined_binary1.png "Binary Example"
[image4]: ./output_images/perspective1.png "straight line perspective example"
[image5]: ./output_images/polyfit1.png "Fit Visual"
[image6]: ./output_images/section_overlayed.jpg "overlayed"
[video1]: ./output_images/out_project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

This document is the project writeup.

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The camera calibration was done using the images provided in the [camera calibration](./camera_cal/) folder. The code for calibration can be found in [calibrate_camera.py](./calibrate_camera.py).

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result. 

![alt text][image1]

The result of the calibration is saved to file for use in the rest of the project.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

Here is an example of a road image before and after distortion correction.
![alt text][image2]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color (S channel in HLS) and gradient (sobel) thresholds to generate a binary image (These operations are performed in functions hls_select and sobel in `road_lines.py`).  Here's an example of my output for this step.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The function get_combined_binary in road_lines.py performs the perspective transform and return the combined binary image after perspective correction.

The parameters for the perspective transform are obtained in [perspective.py](./perspective.py) and stored for use later. The transform is obtained only once. As a result it is done in a separate file.

I verified that my perspective transform was working as expected by drawing the original and perspective image together in a case where the road is straight. The lines appear straight after the perspective transform.

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used two different methods for identifying the pixels corresponding to lanes. The first method uses the histogram of the image in the lower half to identify the position of the line and then uses 12 windows to track the line pixels.

The second methods does not rely on the histogram but rather searches around the previously found lines and if the pixels are within a treshold records them as pixels corresponding to lane lines. If the method looses track of the lines and fails to identify a minimum number of pixels then I revert back to using the histogram method.

After the pixels corresponding to left and right lines are identified, they are fed into a polyfit function to identify the gains of the polynomial. The gains are then filtered to provide smoother curve fits and avoid jumps. The filtering step is essential to make the line detection more robust to shadows and road conditions.

![alt text][image5]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of the curvature for each line is calculated in 'process_image' function in 'road_lines.py' lines 278 and 279 and displayed on the video in real-time in meters.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The function 'process_image' overlays the area between the two lanes on the original image. An example of this is shown below:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/out_project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The implementation of the filter on the gains of the polynomial increased the robustness of the detection. Without the filter, shadows and other road artifact caused the line to jump around. In order to make the algorithm more robust to errors the slope of the two lines can be compared. If a large discrepancy is observed, the history can be used for some amount of time before throwing an error.