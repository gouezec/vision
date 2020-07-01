## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # input shape
        self.conv1 = nn.Conv2d(1, 32, 5)
        # ouput shape: 32 x 220x220 
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.drop1 = nn.Dropout(0.2)
        self.pool1 = nn.MaxPool2d(4)
        # ouput shape: 32 x 55x55 

        self.conv2 = nn.Conv2d(32, 64, 5)
        # ouputshape: 64 x 51x51 
        self.drop2 = nn.Dropout(0.2)
        self.pool2 = nn.MaxPool2d(5)
        # ouput shape: 64 x 10x10 
        
        self.dense3 = nn.Linear(6400, 1000)
        self.drop3 = nn.Dropout(0.2)
        self.dense4 = nn.Linear(1000, 136)
        # ouput shape: 136

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.drop3(F.relu(self.dense3(x)))
        x = self.dense4(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x

		
class Net2(nn.Module):

    def __init__(self):
        super(Net2, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # input shape
        DROP = 0.2
        self.conv1 = nn.Conv2d(1, 32, 5)
        # ouput shape: 32 x 220x220 
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        self.drop1 = nn.Dropout(DROP)
        self.pool1 = nn.MaxPool2d(2)
        # ouput shape: 32 110x110 

        self.conv2 = nn.Conv2d(32, 64, 3)
        # ouputshape: 64 x 108x108
        self.drop2 = nn.Dropout(DROP)
        self.pool2 = nn.MaxPool2d(2)
        # ouput shape: 64 x 54x54 

        self.conv3 = nn.Conv2d(64, 128, 3)
        # ouputshape: 128 x 52x52
        self.drop3 = nn.Dropout(DROP)
        self.pool3 = nn.MaxPool2d(4)
        # ouput shape:  128x 13x13 
	
        self.dense1 = nn.Linear(21632, 1000)
        self.dropd1 = nn.Dropout(DROP)
        self.dense2 = nn.Linear(1000, 136)
        # ouput shape: 136

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(self.pool1(F.relu(self.conv1(x))))
        x = self.drop2(self.pool2(F.relu(self.conv2(x))))
        x = self.drop3(self.pool3(F.relu(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropd1(F.relu(self.dense1(x)))
        x = self.dense2(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
