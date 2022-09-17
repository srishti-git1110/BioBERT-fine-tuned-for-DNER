# A simple perceptron based classifier
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.module):
    def __init__(self, input_size):
        super(Mdel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1)
        
    def forward(self, x, apply_sigmoid=False):
        """Forward pass of the model
        Args:
        x : batch of data, shape should be (batch_size,input_size)
        
        returns an output tensor with shape (batch_size,)"""
        
        y = fc1(x)
        if apply_sigmoid:
            y = F.sigmoid(y)
        return y
