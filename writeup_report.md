# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]:  Results/layerStructure.png "Model Visualization"
[image2]:  Results/CenterDrive.jpg "Grayscaling"
[image3]:  Results/RecoverLeft.jpg "Recovery Image"
[image4]:  Results/RecoverRight.jpg "Recovery Image"
[image5]:  Results/BeforeFlip.png "Recovery Image"
[image6]:  Results/AfterFlip.png "Normal Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I use Keras Sequential model as following
1.Lamda layer is used to normalized the incoming images
2. Cropping2D layer is used to get rid of the noise 
3. I use 4x(Conv+Pooling). Since simple feature has less variance and complex feature has more variance, my feature increases gradually as network go deeper. Each conv layer's feature is 2 time addition of previous layers (2x(f(n-1)+f(n-2))), and first 3 layer filter size is 5x5 and last 2 layers are 3x3 and 1x1 respectivly. Pooling layer just use 2x2 Maxpooling (model.py line55-66)

Each Conv layer is attached with a RELU layer to introduce nonlinearity.

#### 2. Attempts to reduce overfitting in the model

Inorder to reduce overfitting, I augument the data by fliping left and right of the images and also I add both clock-wise and anti-clock wise training data. The returning to the track both from left and right clips are also added in the training data.

During training, I found my training loss (0.0205) and val loss (0.0234) are pretty close, which means it doesn't have big overfitting, that's why the dropout layer are cancelled in the final model. 


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually(model.py 72) .

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and collected both clockwise and anti-clockwise training data 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...



My first step was to use a convolution neural network model similar to the Nvida CNN. so majority of model is consisted of (Conv+RELU+Pooling)x4, then I have 1x1 conv and 2 dense layers. For conv layer, the feature gradually increase because initial layer are trying to capture simple features(lines, dot, circle etc.) where we have less various, but in later layers the feature are more complex, so we need more features channel to capture the as much feature as possible.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I did data augumentation and collected more training data including anti-close wise training data and recovering data.


The final step was to run the simulator to see how well the car was driving around track one. While the data cropping, augmentation and recovering data are not implemented, my car fell off the track near the river and bridge many times. However, with implementation of all of these, my vehicle can successfuly stay on the track all the time ( I kept it running for 2 laps) withour leaving the road. Please see the vidio file.


#### 2. Final Model Architecture

The final model architecture (model.py lines 55-71) are shown as following
(Conv+RELU+Pooling)x4, where Conv filter size are: 3x5x5, 6x5x5, 18x5x5, 48x3x3 respectively and pooling filter are 2x2 Max pooling.
Conv 132x1x1 
Dense 120
Dense 1


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to how to recover to the center of the road. These images show what a recovery looks like starting from left and right:

![alt text][image3]
![alt text][image4]

To augment the data sat, I also flipped images and angles thinking that this would help on the bias to one side turn For example, here is an image that has then been flipped:

![alt text][image5]
![alt text][image6]


After the collection process, I had 9687x2 number of data points. I then preprocessed this data by normalization and cropped top and bottom to remove the unneccesary data.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by min validation loss 0.0234 I used an adam optimizer so that manually training the learning rate wasn't necessary.


