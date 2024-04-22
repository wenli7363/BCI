import torch
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from torch import nn
import torch.nn.functional as F
from torch import optim

class Feature(nn.Module):
    def __init__(self, classes_num):
        super(Feature, self).__init__()
        self.drop_out = 0.25

        # input shape (样本数，1，通道数，时间点数)
        # 修改ZeroPad2d以匹配新的输入尺寸
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((31, 32, 0, 0)),
            nn.Conv2d(
                in_channels=1,  
                out_channels=8,
                kernel_size=(1, 64),
                bias=False
            ),
            nn.BatchNorm2d(8)
        )
        # 经过block1后的输出shape为(样本数，8，通道数，时间点数)

        # 深度可分离卷积层
        self.block_2 = nn.Sequential(
            # input shape (样本数，8，通道数，时间点数)
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(32, 1),
                groups=8,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),  
            nn.Dropout(self.drop_out)
            # output shape (样本数，16，通道数，时间点数//4)
        )

        self.block_3 = nn.Sequential(
            # inoput shape (样本数，16，通道数，时间点数//4)
            nn.ZeroPad2d((7, 8, 0, 0)),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1, 16),
                groups=16,
                bias=False
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=16,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),  
            nn.Dropout(self.drop_out)
            # output shape (样本数，16，通道数，时间点数//32)
        )

        # self.self_attn = SelfAttention(in_channels=16)      # output shape (样本数，16，通道数，时间点数//32)
        # self.out = nn.Linear((992), classes_num)  

    def forward(self, x):
        # print(x.shape)
        x = self.block_1(x)
        # print("经过block1后的x的shape：",x.shape)
        
        x = self.block_2(x)
        # print("经过block2后的x的shape：",x.shape)
        
        x = self.block_3(x)
        # print("经过block3后的x的shape：",x.shape)
        # x = self.self_attn(x)
        # print("经过attention后的x的shape：",x.shape)
        x = x.view(x.size(0), -1)  # 展平特征图以匹配全连接层的输入
 
        # print("展平后的x的shape：",x.shape)
        return x
    
class Predictor1(nn.Module):
    def __init__(self, class_num):
        super(Predictor1, self).__init__()
        self.fc1 = nn.Linear(992, class_num)
        
    def forward(self, x):
        x=self.fc1(x)
        return x

class Predictor2(nn.Module):
    def __init__(self, class_num):
        super(Predictor2, self).__init__()
        self.fc1 = nn.Linear(992, class_num)
        
    def forward(self, x):
        x=self.fc1(x)
        return x


# 注意力机制
class SelfAttention(nn.Module):
    def __init__(self, in_channels):
        super(SelfAttention, self).__init__()
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
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, time)
        
        # 残差连接
        out = self.gamma * out + x
        return out
