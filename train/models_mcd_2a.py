
import torch
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from torch import nn
import torch.nn.functional as F
from torch import optim

# 残差卷积块
# class BasicBlock(nn.Module):
#     expansion = 1
#
#     def __init__(self, inplanes, planes, stride=1, downsample=None):
#         super(BasicBlock, self).__init__()
#         self.conv1 = conv1x1(inplanes, planes)  #+
#         self.bn1 = nn.BatchNorm2d(planes)               #+
#         self.conv2 = conv3x3(planes, planes, stride)
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv3 = conv3x3(planes, planes)
#         self.bn3 = nn.BatchNorm2d(planes)
#         #self.conv4 = conv1x1(planes, planes)
#        # self.bn4 = nn.BatchNorm2d(planes)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         identity = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#        # out = self.relu(out)
#
#        # out = self.conv4(out)
#        # out = self.bn4(out)
#
#         if self.downsample is not None:
#             identity = self.downsample(x)
#
#         out += identity
#         out = self.relu(out)
#
#         return out

# 特征提取器
class Feature(nn.Module):
    def __init__(self, classes_num):
        super(Feature, self).__init__()
        self.drop_out = 0.25

        self.block_1 = nn.Sequential(
            # Pads the input tensor boundaries with zero
            # left, right, up, bottom
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,  # input shape (1, C, T)
                out_channels=8,  # num_filters
                kernel_size=(1, 64),  # filter size
                bias=False
            ),  # output shape (8, C, T)
            nn.BatchNorm2d(8)  # output shape (8, C, T)
        )

        # block 2 and 3 are implementations of Depthwise Convolution and Separable Convolution
        # 深度可分离卷积
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,  # input shape (8, C, T)
                out_channels=16,  # num_filters
                kernel_size=(22, 1),  # filter size
                groups=8,
                bias=False
            ),  # output shape (16, 1, T)
            nn.BatchNorm2d(16),  # output shape (16, 1, T)
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  # output shape (16, 1, T//4)
            nn.Dropout(self.drop_out)  # output shape (16, 1, T//4)
        )

        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 16),  # filter size
                groups=16,
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)
            nn.Dropout(self.drop_out)
        )

        self.out = nn.Linear((496), classes_num)

    def forward(self, x):
        x = self.block_1(x)
        # print("block1", x.shape)
        x = self.block_2(x)
        # print("block2", x.shape)
        x = self.block_3(x)
        # print("block3", x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        
        return  x  # return x for visualization






class Predictor1(nn.Module):
    def __init__(self, class_num):
        super(Predictor1, self).__init__()
        self.fc1 = nn.Linear(496, class_num)
        
    def forward(self, x):
       
        x=self.fc1(x)
        return x

class Predictor2(nn.Module):
    def __init__(self, class_num):
        super(Predictor2, self).__init__()
        self.fc1 = nn.Linear(496, class_num)
        
    def forward(self, x):
       
        x=self.fc1(x)
        return x



#
# data_loaders = dataloader2.dataloader(batch_size=256, type=3, ratio=8,method=None)
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#
#
#
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.003)
# train_model.train(criterion=criterion, change=False, dataloaders=data_loaders, device=device, model=net,
#                   optimizer=optimizer, num_epochs=50)

# f=Feature()
# c=Predictor()
# a=torch.randn(128,1,2400)
# a=f(a)
# a=c(a)
# print(a.shape)
#
# net=Net()
# data_loaders = dataloaders_p.dataloaders_p(file_type=3)
# device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(net.parameters(), lr=0.005)
# train_model.train(criterion=criterion, change=False, dataloaders=data_loaders, device=device, model=net,
#                   optimizer=optimizer, num_epochs=10)
