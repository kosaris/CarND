# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/model_plot.png "Model Architecture"
[image2]: ./examples/centerlane_driving.jpg "center driving"
[image3]: ./examples/recovery1.jpg "recovery1"
[image4]: ./examples/recovery2.jpg "recovery2"
[image5]: ./examples/recovery3.jpg "recovery3"
[image6]: ./examples/loss.png "loss"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* train.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The train.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model utilizes the NVIDIA neural network with multiple convolution and fully connected layers.

The model includes RELU layers to introduce nonlinearity in its activation later, and the data is normalized in the model using a Keras lambda layer.

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in each of its fully connected layers in order to reduce overfitting. (dropout late = 0.5)

Dropout layers, in combination with larger size data collection help avoid overfitting to the training data. The validation loss and training loss is shown in the figure below.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually. Dropout rate was tuned manually by trying out a few numbers. In addition, the number of epochs were tuned after observing that epochs beyond 5 do not provide any meaningful improvement in results (and might even overfit the training set).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, pointing the car to the edges and then turning back. During the recovery recordings, only the recovery section was recorded. The part that sends the car to the edge was not recorded. To avoid the bias, the car was driven both ways around the track. An optical mouse was used to steer the car around the track and resulted in significantly better training data compared to keyboard input.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to use a network with a number of convolution layers and fully connected layers.

My first step was to use a convolution neural network model similar to the NVIDIA network. This type of network was used to the traffic sign classifier. This model seems to be appropriate since the convolution layers can convert the images into its features (edges, lines, shapes, etc.) and along with the fully connected layer can estimate steering angles.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (20%- 80% respectively). 

To combat the overfitting, I modified the model to add dropout in the fully connected layers. Additional data was also collected to avoid overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. I collected more data around those areas to teach the model to recover in these situations.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted of a convolution neural network with the layers shown below:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving and then two laps in the reverse direction to avoid the left-steering bias. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

To augment the data sat, I also flipped images and angles thinking that this would help minimize overfitting and will generalize the network. I also used the left and right camera images with a correction factor of 0.2.


The image was also cropped to contain as much as the road section as possible.

After the collection process, I had 44784 number of data points. I used the generator but it did not speed up the model. Therefore I went back to using the model without generator and loading the data points in the memory. Training was done on a GPU.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

The loss in the trainig set and the validation set is shown in the figure below. Training more epochs did not improve the performance.
![alt text][image6]
