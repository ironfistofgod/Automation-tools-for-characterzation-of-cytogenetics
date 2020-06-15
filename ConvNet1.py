
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class MyConvNet(nn.Module):
    def __init__(self):
        super(MyConvNet, self).__init__()

        self.relu = nn.ReLU()
        self.dropout03 = nn.Dropout(0.3)
        self.dropout04 = nn.Dropout(0.4)

        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0)
        self.bn5 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
        self.bn7 = nn.BatchNorm2d(512)
        self.pool7 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0)
        self.bn8 = nn.BatchNorm2d(256)
        self.pool8 = nn.MaxPool2d(kernel_size=2, stride=2)
       

        self.fc1 = nn.Linear(512*2*3, 5)
        self.fc2 = nn.Linear(1500, 256)
        self.fc3 = nn.Linear(512, 5)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu(self.bn4(torch.nn.Dropout(0.2)(self.conv4(x)))))
        x = self.pool5(self.relu(self.bn5(self.conv5(x))))
        x = self.pool6(self.relu(self.bn6(torch.nn.Dropout(0.2)(self.conv6(x)))))
        x = self.pool7(self.relu(self.bn7(self.conv7(x))))
#        print(x.shape)
# print('pool7', x.shape)#

#        x = self.pool8(self.relu(self.bn8(self.conv8(x))))
#        print('xxxxxxxxxxxxxxxxxx', x.shape)
        x = x.view(-1, 512*x.shape[2]*x.shape[3])
#        print('view', x.shape)
        x = self.fc1(x)
#        print('x', x.shape)
#        x = (self.fc2(x))
#        x = self.fc3(x)
#        return self.softmax(x)
#        print(self.fc1(x))
        return x
