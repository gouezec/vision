## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        DROP = 0.2        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # input shape
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        self.features = nn.Sequential(
                            nn.Conv2d(1,32,5), #32 x 220x220 
                            nn.ReLU(),
                            nn.MaxPool2d(4),   #32 x 55x55
                            nn.Dropout(DROP),

                            nn.Conv2d(32,64,5),#64 x 51x51
                            nn.ReLU(),
                            nn.MaxPool2d(5),   #64 x 10x10
                            nn.Dropout(DROP))

        self.regression = nn.Sequential(     
                            nn.Linear(64*10*10, 1000),
                            nn.ReLU(),
                            nn.Dropout(DROP),
                            nn.Linear(1000, 136))
        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regression(x)
        
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
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting

        self.features = nn.Sequential(         #1 x 224x224 
                            nn.Conv2d(1,32,5), #32 x 220x220 
                            nn.ReLU(),
                            nn.MaxPool2d(3),   #32 x 73x73
                            nn.Dropout(DROP),

                            nn.Conv2d(32,64,4),#64 x 70x70
                            nn.ReLU(),
                            nn.MaxPool2d(3),   #64 x 23x23
                            nn.Dropout(DROP),

                            nn.Conv2d(64,128,3),#128 x 21x21
                            nn.ReLU(),
                            nn.MaxPool2d(3),   #128 x 7x7
                            nn.Dropout(DROP))


        self.regression = nn.Sequential(     
                            nn.Linear(128*7*7, 1000),
                            nn.ReLU(),
                            nn.Dropout(DROP),
                            nn.Linear(1000, 136))

    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regression(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x



class Net3(nn.Module):

    def __init__(self):
        super(Net3, self).__init__()
	self.features = nn.Sequential( 
        			nn.Conv2d(1, 32, 5),	# 1 x 220x220
        			nn.MaxPool2d(2, 2),	# 32 
        		        nn.Dropout(0.1),
        			nn.Conv2d(32, 64, 3),	#110--> 110-3+1=108
        			nn.MaxPool2d(2, 2),     #108/2=54
        			nn.Dropout(0.2),
        			nn.Conv2d(64, 128, 3),	#54-->54-3+1=52
        			nn.MaxPool2d(2, 2),	#52/2=26
        			nn.Dropout(0.3),
				nn.Conv2d(128, 256, 3),	#26-->26-3+1=24
    				nn.MaxPool2d(2, 2),	#24/2=12
        			nn.Dropout(0.4),
        			nn.Conv2d(256,512,1),	#12-->12-1+1=12
        			nn.MaxPool2d(2,2),	#12/2=6
		                nn.Dropout(0.5))
        
	self.regression = nn.Sequential(     
                            nn.Linear(512*6*6, 1024),
                            nn.ReLU(),
                            nn.Dropout(0.6),
                            nn.Linear(1024, 136))
		        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.regression(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
