#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 3 and 64 (model.py lines 66-77) 

The model includes RELU layers to introduce nonlinearity (code line 66-77), and the data is normalized in the model using a Keras lambda layer (code line 64). 

####2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 66-77). 

However if I used drop outs, the car was not driving well in the track and the val error was not high. Hence a drop out rate of 1 was used.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 92).

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I tried the following:
a) Driving in center lane
b) Driving with zig zags continously
c) Driving with just recovering from sides to center
d) Driving in reverse order.

After lots of testing, it was found that a simple one lap of center lane driving did the trick.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to start with a simple architecture and transforming it to a complicated network

My first step was to use a convolution neural network model similar to the Lanet. I thought this model might be appropriate because it is a simple model. Then I used the architecture given in project description to train to my data

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set and a low mean squared error on the validation set. 

Since there was no over fitting I did not apply drop outs.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I kept retraining the model with different data sets as given in #4 Training data.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 66-77) consisted of a convolution neural network. Please refer model.py for more details.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track one using center lane driving. 



Etc ....

After the collection process, I had 1221 number of data points. I then preprocessed this data by ...

a) splitting into training and val sets.
b) Writing a generator with batch size 32
c) Normalized every image around 127 with standard deviation of 0.5
d) Randomly selecting center image, left image and right image
e) If left or right image was used, inserted a steering offset of 0.35
f) Randomly flipping images so that the training set is not concentrated on left side

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 15. I used an adam optimizer so that manually training the learning rate wasn't necessary.
