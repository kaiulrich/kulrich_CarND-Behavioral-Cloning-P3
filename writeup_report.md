#**Behavioral Cloning** 


[//]: # (Image References)

[architecture]: ./images/architecture.png "Model Visualization"
[tensorboard]: ./images/Tensorbord.png "Tensorboard"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### I Projectstrucktur
-----

#### 1. My project includes the following files:

* model.py containing the script to create and train the model
* network.py containing the script to define the Keras Network
* generator.py containing the script for the train and validate data generator 
* utils.py containing the script for the loading data and preprozess images
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* CarND-Behavioral-Cloning-P3.ipynb notebook to generate documentation images
* writeup_report.md summarizing the results

####2. Run model on simulatror
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

####3. Code for training the model

The network.py file contains the code for training and saving the convolution neural network. 

1.  **Before loading the data, the header of the csv File has to be deleted !!** 
2. Data the set is splited into trainingdata
3. Model is created. 
4. The train and validation generators are created.
5. AdamOptimizer is created
6. Compile the model
7. Run training via fit_generator method.
8. After the training we have to analyse which Epic produced the best model. it can be find in the models folder.
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
![alt text][tensorboard]
###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model is a variant of the NVIDIA model which is a normalisation layer followed by 5 convolution layers and 3 fully connected layers. My model starts with a cropping layer and a normalisation layer. This could be done in the preprocessing shape but it seems more readable and probably more efficient (use of GPU in training) to put it in the ANN. There althought is no preprozessing in the drive.py nessesery. In the training preprocessing pipline, I load one of the 3 camera angle at random with a shift ajustedment to account for the left and right camera offset. I also augment the data at random by flip, brighness, dark shadows, shears to desensitise the model to lighting conditions, balance the road topology. 



####2. Attempts to reduce overfitting in the model

The data are splited into a training and validation data set.
The training and validation data are randomly selected from generators. and the model contains dropout layers in order to reduce overfitting. The first dropout layers is right after the three Convolution layers (network.py lines 27). The next dropout layers are after the first (network.py lines 31) and second (network.py lines 27) fully connected layers.

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track 1 and track 2.

####3. Model parameter tuning

For the loss function I used MSE and used the ADAM optimiser with a low initial learning rate (model.py line 72).. The Adam optimizer as compared to SGD automatically adjusts the learning rate over epochs unlike SGD.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][architecture]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
