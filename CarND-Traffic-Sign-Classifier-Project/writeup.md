# **Traffic Sign Recognition** 
---

**Build a Traffic Sign Recognition Project**

The steps of this project are the following:

* Load the data set and augment images (rotate, zoom, etc.)
* Explore and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)
[image3]: ./examples/failures2.jpeg "misses"
[image4]: ./examples/exploratoru_visualization.png "exploratory_visualization"
[image5]: ./examples/augmentation.png "augmentation"
[image6]: ./examples/architecture.png "architecture"
[image7]: ./examples/five_downloaded1.png "five downloaded"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./examples/placeholder.png "Traffic Sign 5"
[image10]: ./examples/placeholder.png "Traffic Sign 5"
[image11]: ./examples/placeholder.png "Traffic Sign 5"
[image12]: ./examples/placeholder.png "Traffic Sign 5"
[image13]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

This document is the write-up. Here is a link to my [project code](https://github.com/kosaris/CarND/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python/numpy to calculate summary statistics of the traffic
signs data set:

Before augmentation:

* number of examples in training dataset: 34799
* number of examples in validation dataset: 4410
* number     of examples in testing dataset: 12630
* size of data: (32, 32, 3)
* number of classes: 42

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. I have plotted 16 images along with their labels below.

![alt text][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale. The information from the R, G and B channels are averaged in making the grayscale image. This reduces the size of the network (32x32x1 instead of 32x32x3) and reduces the dimension of the network. In addition, it takes care of offsets in color.


I decided to generate additional data using data augmentation. A zoomed in traffic sign is still the same traffic sign. So is a rotated traffic sign. The augmented data set provides more training data without the need to do more costly data collection and makes the network more robust as it exposes it to more variation.

The following functions are applied on each of the original images to create 4 augmented images:

* Random rotation
* Random shear
* Random shift
* Random zoom

Here is an example of the augmented images

![alt text][image5]

Finally I normalize the images by subtracting 128 from the data and dividing by 128. This normalization ensures that the data has similar distribution.

After augmentation:

* number of examples in training dataset: 173995
* number of examples in validation dataset: 4410
* number     of examples in testing dataset: 12630
* size of data is (32, 32, 3)
* number of classes is 42

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The architecture of the network is shown below:
 
![alt text][image6]

2 convolution layers, 2 pooling layers and 3 fully connected layers are used.

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The mean of the cross entropy was choses as the cost function. L2 regularization (with beta of .2) was used to keep the gains from growing out of bound and making the network more robust. In addition, dropout (with keep_prob of 52%) was applied. 

The batch size was set to 128 and the learning rate of 0.0003 was used with 400 epochs.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 95.0%
* validation set accuracy of 94.4% 
* test set accuracy of 92.1%

As the initial architecture I applied LeNet directly. This got me to validation accuracy of 89% but the network had a large bias as the training accuracy was close to 98%. Increasing the size of the network would not help in this situation. 

I increased the drop out rate to reduce the memorization and overfitting. This was helpful and got the validation accuracy slightly higher but not enough. 

After some trial and error with learning rate and other parameters, I tired data augmentation. This had the largest impact and was able to bring the training accuracy down and increase the validation accuracy. The network was no longer overfitting as much.

L2 regularization was the next thing I tried and it helped to further reduce the bias problem. Now that the network was no longer overfitting, I increased the size of the network slightly to increase validation accuracy.

The above combination gave me the accuracy of over 93% in validation set. The training accuracy was . This means that a larger network can possibly help increase the training and validation accuracy even further.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 6 images that I downloaded from the web:

![alt text][image7]

Their true value (t:) and their predicted label (p:) is shown above each image.

The 2nd image is difficult because of the angle at which it is taken. The 4th image is stretched horizontally and is taken from below and the stop sign image is taken from the side. The rest of the images have other lines and background objects in them that might make prediction difficult.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Turn right ahead              		| Turn right ahead   		|  
| Right-of-way at the next intersection | Right-of-way at the next intersection 										                |
| Road work         					| Road work	                |
| Priority road	     	            	| Priority road				|
| Wild animals crossing	        		| Wild animals crossing     |
| Stop		                        	| Stop      				|


The model was able to correctly guess all six traffic signs, which gives an accuracy of 100%. The accuracy of the test set was 92.1%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The model predicts the images with very high certainty. Only one of the probabilities is set to 1.0 and the rest are at 0.0.

The following is the print of top 5 probabilities for the first image.

TopKV2(values=array([[1., 0., 0., 0., 0.]], dtype=float32), indices=array([[11,  0,  1,  2,  3]], dtype=int32))

To test that I am not printing the wrong value, I downloaded a cat image and passes it through the network. The following are the top five probabilities for the cat image:

[[1.0000000e+00, 1.0559258e-20, 0.0000000e+00, 0.0000000e+00,
        0.0000000e+00]]

Logits are defined by:
```
logits = tf.nn.softmax(tf.add(tf.matmul(fc3, wd4), bd4))
```

#### 4. Miss-labeled Images
The following images were misclassified. Most of the misclassified images are due to very poor lighting conditions.
![alt text][image3]{:height="560px"}
