
import torch
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from torch import nn
import torch.nn.functional as F
from torch import optim
import math


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
            ),  # output shape (16, 1, T//4)  因为有padding，用的same模式卷积
            nn.Conv2d(
                in_channels=16,  # input shape (16, 1, T//4)
                out_channels=16,  # num_filters
                kernel_size=(1, 1),  # filter size
                bias=False
            ),  # output shape (16, 1, T//4)
            nn.BatchNorm2d(16),  # output shape (16, 1, T//4)
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  # output shape (16, 1, T//32)        # 池化，再除以8
            nn.Dropout(self.drop_out)
        )
        self.self_attn = SelfAttention(in_channels=16)
        # self.out = nn.Linear((496), classes_num)          # feature并不需要全连接层，因为后面还有两个分类器

    """
    torch.Size([128, 1, 22, 1000])
    经过block1:  torch.Size([128, 8, 22, 1000])
    经过block2:  torch.Size([128, 16, 1, 250])
    经过block3:  torch.Size([128, 16, 1, 31])
    经过attention:  torch.Size([128, 16, 31])
    torch.Size([128, 496])
    """
    def forward(self, x):
        # print(x.shape)
        x = self.block_1(x)
        # print("经过block1: ",x.shape)
        # print("block1", x.shape)
        x = self.block_2(x)
        # print("经过block2: ",x.shape)
        # print("block2", x.shape)
        x = self.block_3(x)
        # print("block3", x.shape)
        # print("经过block3: ",x.shape)
        x = self.self_attn(x)
        # print("经过attention: ",x.shape)
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


# 注意力机制
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
        # 输出通道缩小8倍
        self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.key = nn.Conv1d(in_channels, in_channels // 8, 1)
        self.value = nn.Conv1d(in_channels, in_channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size = x.size(0)
        
        # 合并非批次维度
        x = x.view(batch_size, -1, x.size(-1))
        channels, time = x.size(1), x.size(2)
        
        # 计算 query, key, value
        proj_query = self.query(x).view(batch_size, -1, time).permute(0, 2, 1)
        proj_key = self.key(x).view(batch_size, -1, time)
        proj_value = self.value(x).view(batch_size, -1, time)
        
        # 计算注意力权重
        energy = torch.bmm(proj_query, proj_key)
        attention = torch.softmax(energy, dim=-1)
        
        # 注意力加权
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))     # permute对维度进行转置
        out = out.view(batch_size, channels, time)
        
        # 残差连接
        out = self.gamma * out + x
        return out
