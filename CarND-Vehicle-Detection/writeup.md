**Vehicle Detection Project**

The goals / steps of this project are the following:

* Use Histogram of Gradients (HOG) features, Color histogram features and binned color features extracted from a set of vehicle and non-vehicle images to train a linear SVM classifier.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/HOG_example.png
[image2]: ./output_images/vehicle_detection2.png
[image3]: ./output_images/vehicle_detection3.png
[image4]: ./output_images/vehicle_detection4.png
[image5]: ./output_images/vehicle_detection5.png
[image6]: ./output_images/heat_map1.png
[image7]: ./output_images/boxes1.png


## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

This writeup is the report.

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The `get_hog_features` function extracts the HOG features. This can be found in `xxxxx.py`. This uses the `skimage.hog()` function.

#### 2. Explain how you settled on your final choice of HOG parameters.

I played with the various arguments of the function and used the prediction accuracy of the classifier to choose the optimal choice. I settled on the following parameters for feature selection:

```
orient = 9
pix_per_cell = 8
cell_per_block = 2
```
Also after playing with different HOG channels I decided to use all channels for HOG feature extraction as opposed to a single channel.

![alt text][image1]

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The classifier is a Linear SVM. It is trained in `classifier_training.py`. I played with various features with the goal of maximizing the accuracy of the classifier and settled on using all features (Histogram of Gradients (HOG), Color histogram and binned color). The classifier yields a test accuracy of 98.56% on the training data.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to use various scales at different areas of the image. I used window sizes of 64, 96, 128 and 196 pixels (two for each size). The overlap was selected to be 75%. Better results are achieved with higher overlap, however, it increases the execution time.

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image2]
![alt text][image3]
![alt text][image4]
![alt text][image5]

The pipeline is able to identify the vehicles with acceptable false positives.
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used multiple size boxes on every frame to detect the vehicles and created a heatmap for each pixel that falls in any of the boxes. I then applied a threshold on the heatmap with the assumption that the false positives do not appear in multiple boxes but the car images will. I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap. I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here is a frame with its heatmap:
![alt text][image6]


### Here some bounding boxes are drawn on a frame
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main area for improvement is to reduce false positive and increase the success rate. One way is to use a CNN instead of SVM for detection. We can also employ a temporal heatmap assuming that a vehicle will not suddenly appear or disappear. The heatmap can be extended over multiple frames.

The second area of improvement is to improve the speed. One way to do this is to limit the search area by using the history of detection. A vehicle will enter into the frames from certain areas. This way the search area can be limited to near the previous detection as well as near the edges. 

