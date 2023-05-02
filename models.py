
import torch 
from torch import nn

class CNN(torch.nn.Module):
    def __init__(self, in_channels=20, output_size=1):
        super().__init__()
        self.conv1 = nn.Sequential(
        	nn.ConstantPad1d((2,1), 0),
            nn.Conv1d(in_channels=in_channels, 
            	out_channels=128, 
            	kernel_size=4, 
            	stride=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10)
        )
        # self.conv2 = nn.Sequential(
        #     nn.ConstantPad1d((2,1), 0),
        #     nn.Conv1d(in_channels=128, 
        #         out_channels=64, 
        #         kernel_size=4, 
        #         stride=1),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.MaxPool1d(10)
        # )
        self.flatten = nn.Flatten()
        self.layer2 = nn.Sequential(
        	nn.Linear(8192, 128),
        	nn.ReLU()
        )
        self.layer3 = nn.Sequential(
        	nn.Linear(128, 2),
        	nn.Softmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

class DeepCNNAcharya(torch.nn.Module):
    def __init__(self, in_channels=20, output_size=1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConstantPad1d((3,2),0),
            nn.Conv1d(in_channels=in_channels,
                out_channels=4096, 
                kernel_size=6, 
                stride=1),
            nn.ReLU())
        self.maxpool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((2,2),0),
            nn.Conv1d(in_channels=4096,
                out_channels=2048, 
                kernel_size=5, 
                stride=1),
            nn.ReLU())
        self.maxpool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((2,1),0),
            nn.Conv1d(in_channels=2048,
                out_channels=1024, 
                kernel_size=4, 
                stride=1),
            nn.ReLU())
        self.maxpool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv4 = nn.Sequential(
            nn.ConstantPad1d((1,2),0),
            nn.Conv1d(in_channels=1024,
                out_channels=512, 
                kernel_size=4, 
                stride=1),
            nn.ReLU())
        self.maxpool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5 = nn.Sequential(
            nn.ConstantPad1d((2,1),0),
            nn.Conv1d(in_channels=512,
                out_channels=256, 
                kernel_size=4, 
                stride=1),
            nn.ReLU())
        self.maxpool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(5120, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 2)
        self.out = nn.Softmax()
        self.layers = [self.conv1, self.maxpool1, 
                       self.conv2, self.maxpool2,
                       self.conv3, self.maxpool3,
                       self.conv4, self.maxpool4,
                       self.conv5, self.maxpool5,
                       self.flatten,
                       self.fc1, self.fc2, self.fc3, self.out]


    def forward(self, x):
        for l in self.layers:
            x = l(x)
        return x





