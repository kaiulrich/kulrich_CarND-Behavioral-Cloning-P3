# Behavioral Cloning


[//]: # (Image References)

[architecture]: ./images/architecture.png "Model Visualization"
[tensorboard]: ./images/Tensorbord.png "Tensorboard"
[histogram]: ./images/histogram.png "Histogram"

[camara_right]: ./images/camara_right.jpg "Camara right"
[camara_center]: ./images/camara_center.jpg "Camara center"
[camara_left]: ./images/camara_left.jpg "Camara left"


[pre_org]: ./images/orig.png "Original Image"
[pre_fliped]: ./images/fliped.png "Flipped Image"
[pre_brightness]: ./images/brightness.png "Brightness Image"
[pre_sharing]: ./images/share.png "Sharing Image"
[pre_shadowing]: ./images/shadow.png "Shadowing Image"
[pre_all_pipeline]: ./images/all_pipeline.png "Piplined Image"

[netw_nvida]: ./images/nvida_network.png "Nvida_network" 

The goals / steps of this project are the following:

* Use the simulator to collect data of good driving behavior 
* Design, train and validate a model that predicts a steering angle from image data
* Use the model to drive the vehicle autonomously around the first track in the simulator. The vehicle should remain on the road for an entire loop around the track.

----
###  Projectstrucktur

#### 1. My project includes the following files:

* model.py containing the script to create and train the model
* network.py containing the script to define the Keras Network
* generator.py containing the script for the train and validate data generator 
* utils.py containing the script for the loading data and preprozess images
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* CarND-Behavioral-Cloning-P3.ipynb notebook to generate documentation images
* RREADME.md summarizing the results

#### 2. Run model on simulatror
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

#### 3. Code for training the model

The network.py file contains the code for training and saving the convolution neural network. 

1.  **Before loading the data, the header of the csv File has to be deleted !!** 
2. Data the set is splited into training and validation data
3. Model is created. 
4. The train and validation generators are created.
5. AdamOptimizer is created
6. Compile the model
7. Run training via fit_generator method.
8. After the training we have to analyse which Epic produced the model with the lowest loos.  It can be find in the models folder.
We can do this via the training console logs
 
```sh
Epoch 1/20
10000/10000 [==============================] - 1906s - loss: 0.0553 - val_loss: 0.0299
Epoch 2/20
10000/10000 [==============================] - 1900s - loss: 0.0373 - val_loss: 0.0289
Epoch 3/20
10000/10000 [==============================] - 1898s - loss: 0.0340 - val_loss: 0.0280
Epoch 4/20
10000/10000 [==============================] - 1900s - loss: 0.0326 - val_loss: 0.0271
Epoch 5/20
10000/10000 [==============================] - 1897s - loss: 0.0314 - val_loss: 0.0273
Epoch 6/20
10000/10000 [==============================] - 1898s - loss: 0.0307 - val_loss: 0.0265
Epoch 7/20
10000/10000 [==============================] - 1899s - loss: 0.0300 - val_loss: 0.0260
Epoch 8/20
10000/10000 [==============================] - 1900s - loss: 0.0295 - val_loss: 0.0256
Epoch 9/20
10000/10000 [==============================] - 1899s - loss: 0.0289 - val_loss: 0.0256
Epoch 10/20
10000/10000 [==============================] - 1899s - loss: 0.0285 - val_loss: 0.0251
Epoch 11/20
10000/10000 [==============================] - 1900s - loss: 0.0282 - val_loss: 0.0256
Epoch 12/20
10000/10000 [==============================] - 1897s - loss: 0.0277 - val_loss: 0.0256
```
or running the tensor bord
```sh
 tensorboard --logdir=logs
```
----- 
### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is a variant of the NVIDIA model which is a normalisation layer followed by 5 convolution layers and 3 fully connected layers. My model starts with a cropping layer. This could be done in the preprocessing shape but it seems more readable and probably more efficient (use of GPU in training) to put it in the ANN. There althought is no preprozessing in the drive.py nessesery. In the training preprocessing pipline, I load one of the 3 camera angle at random with a shift ajustedment to account for the left and right camera offset. I also augment the data at random by flip, brighness, dark shadows, shears to desensitise the model to lighting conditions, balance the road topology. 


#### 2. Attempts to reduce overfitting in the model

The data are splited into a training and validation data set.
The training and validation data are randomly selected from generators. and the model contains dropout layers in order to reduce overfitting. The first dropout layers is right after the three Convolution layers (network.py lines 27). The next dropout layers are after the first (network.py lines 31) and second (network.py lines 27) fully connected layers.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track 1 and track 2.

#### 3. Model parameter tuning

For the loss function I used MSE and used the ADAM optimiser with a low initial learning rate (model.py line 72). The Adam optimizer as compared to SGD automatically adjusts the learning rate over epochs unlike SGD. Because of the random train and validation se it is possible to variate the Numbers of images by batch. 

#### 4. Appropriate training data

I used the images preperated for this project. Thanx to my son ( who is a lot better driver than I am) we produced around 4000 aditional data rows. 

 ![alt text][histogram]
 
The of the training data histogram shows, that there the most data are are from small steering angles. I decided not to clearing them up to have a real cloning of the behaviour.

To run the training on usal machines i desided to implement a validation and training data genererator on each data set. The training data generator selects randomly Images from center/left/right cameras. The main purpose of the images form left/right cameras is to learn model how to recover when the vehicle drives off the center the lane. Small value 0.3 has been added to the steering angle for left camera and subtracted form the steering angle for right camera. 

| Camara left                    |      Camara center               |       Camara right                        | 
|--------------------------------|------------------------------------|------------------------------------------|
| ![alt text][camara_left] | ![alt text][camara_center]  |  ![alt text][camara_right]          | 

Also each image and corresponding steering angle has been flipped and added to the training set. This helps to balance data because original training data contains more turns to the left.

| Original                     | Fliped                       |
|----------------------------|---------------------------|
| ![alt text][pre_org]   | ![alt text][pre_fliped] |

The Training data generator does some Image preprozessing too, like changing the brightness, image sharing or shadowing


| Brightness                 				 | Sharing                       			  |  Shadowing      			             | 
|---------------------------------------------------|---------------------------------------------|----------------------------------------------|
| ![alt text][pre_brightness]                     | ![alt text][pre_sharing]                  |  ![alt text][pre_shadowing]            | 


Mutch later I found the [ImageDataGenerator](https://keras.io/preprocessing/image/) provided the same work for the same preprozessings.


----- 
### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I started with NVIDIA model with 5 convolution layers and 4 fully connected layers (see image below) I reduced the complexity because we dont have the same complexity.
![alt text][netw_nvida]
I continued with 3 convolution layers and 3 fully connected layers and internal Cropping2D (to cut the area of interesst) and a Normalisation layer. 

At the end of the process, the vehicle is able to drive autonomously around the track 1 and Track 2 without leaving the road.

#### 2. Final Model Architecture

The final model architecture (see picture below) consisted of a convolution neural network with the following layers and layer sizes. 

![alt text][architecture]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving and added the tem to the prepared training data.
Than put 20% of the data into a validation set. 

I used 20000 randomly taken flipped and preprozessed training data for training the model. 

Example for a preprozessing pipline result

| Original                     | After pipline                       |
|----------------------------|---------------------------|
| ![alt text][pre_org]   | ![alt text][pre_all_pipeline] |

The validation set helped determine if the model was over or under fitting data. The Keras EarlyStoppin Callback (max epoch 20) was used to stop the training when the validation los hit the bottom value and starts rising agin.
The ideal number of epochs was 8 ore 9. 

![alt text][tensorboard]

I was using an adam optimizer so that manually training the learning rate wasn't necessary.

After I fond my best model, I started to edit the speed parameter in the drive.py (line 48) and I figured out the the best result in the 2 Track is 13

----- 
### Result 

The result model starts a bit curvy on track one but gets smover after awile. It behaves even better on track 2. 

| Track 1                     | Track 2                       |
|---------------------------|---------------------------|
[![E](https://img.youtube.com/vi/senH6s-iNyQ/0.jpg)](https://youtu.be/senH6s-iNyQ "Training Track - Track 1") | [![E](https://img.youtube.com/vi/x6gXYwm-jrE/0.jpg)](https://youtu.be/x6gXYwm-jrE "Training Track - Track 2")|

----- 
### Review

The vehicle is able to drive autonomously around the track 1 and Track 2 without leaving the road. It would be interessting to train the other features 'throttle', 'brake', 'speed' as well.

An improvment coul be to train the network with more balanced data like clearing up the data to loss the strong weight of small steering angles. 

The good result from track 2 surised me, because I did not use any traing data form that track.

karas provides a lot of tools worth to discover :-)

