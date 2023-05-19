
# cnn_1
class CNN(torch.nn.Module):
    def __init__(self, in_channels=20):
        super().__init__()
        self.conv1 = nn.Sequential(
        	nn.ConstantPad1d((2,1), 0),
            nn.Conv1d(in_channels=in_channels, 
            	out_channels=1024, 
            	kernel_size=4, 
            	stride=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10)
        )
        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((2,1), 0),
            nn.Conv1d(in_channels=1024, 
                out_channels=512, 
                kernel_size=4, 
                stride=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10)
        )
        self.flatten = nn.Flatten()
        self.layer2 = nn.Sequential(
        	nn.Linear(3072, 128),
        	nn.ReLU()
        )
        self.layer3 = nn.Sequential(
        	nn.Linear(128, 2),
        	# nn.Softmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# cnn_2
class CNN(torch.nn.Module):
    def __init__(self, in_channels=20):
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
        #     nn.Conv1d(in_channels=1024, 
        #         out_channels=512, 
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
            # nn.Softmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.flatten(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# cnn_3
class CNN(torch.nn.Module):
    def __init__(self, in_channels=20):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConstantPad1d((2,1), 0),
            nn.Conv1d(in_channels=in_channels, 
                out_channels=1024, 
                kernel_size=4, 
                stride=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10)
        )
        # self.conv2 = nn.Sequential(
        #     nn.ConstantPad1d((2,1), 0),
        #     nn.Conv1d(in_channels=1024, 
        #         out_channels=512, 
        #         kernel_size=4, 
        #         stride=1),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.MaxPool1d(10)
        # )
        self.flatten = nn.Flatten()
        self.layer2 = nn.Sequential(
            nn.Linear(65536, 128),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128, 2),
            # nn.Softmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        # x = self.conv2(x)
        x = self.flatten(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# cnn_4 eval auroc 0.87
class CNN(torch.nn.Module):
    def __init__(self, in_channels=20):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConstantPad1d((2,1), 0),
            nn.Conv1d(in_channels=in_channels, 
                out_channels=512, 
                kernel_size=4, 
                stride=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10)
        )
        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((2,1), 0),
            nn.Conv1d(in_channels=512, 
                out_channels=256, 
                kernel_size=4, 
                stride=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10)
        )
        self.flatten = nn.Flatten()
        self.layer2 = nn.Sequential(
            nn.Linear(65536, 128),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128, 2),
            # nn.Softmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# cnn_5
class CNN(torch.nn.Module):
    def __init__(self, in_channels=20):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ConstantPad1d((2,1), 0),
            nn.Conv1d(in_channels=in_channels, 
                out_channels=1024, 
                kernel_size=4, 
                stride=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10)
        )
        self.conv2 = nn.Sequential(
            nn.ConstantPad1d((2,1), 0),
            nn.Conv1d(in_channels=1024, 
                out_channels=512, 
                kernel_size=4, 
                stride=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(10)
        )
        self.conv3 = nn.Sequential(
            nn.ConstantPad1d((2,1), 0),
            nn.Conv1d(in_channels=512, 
                out_channels=256, 
                kernel_size=4, 
                stride=1),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.MaxPool1d(3)
        )
        self.flatten = nn.Flatten()
        self.layer2 = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Linear(128, 2),
            # nn.Softmax()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


# cnn_6 eval auroc ___


