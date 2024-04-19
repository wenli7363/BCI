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

        self.block_1 = nn.Sequential(
            # 将输入尺寸修改为(32, 2000)
            nn.ZeroPad2d((31, 31, 0, 0)),  # 在时间维度上进行零填充
            nn.Conv2d(
                in_channels=1,  # 输入通道数为1
                out_channels=8,  # 输出通道数为8
                kernel_size=(1, 64),  # 卷积核尺寸为(1, 64)
                bias=False
            ),  # 输出尺寸为(8, 32, 1969)
            nn.BatchNorm2d(8)  # 批归一化
        )

        # block 2 和 3 保持不变
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(22, 1),
                groups=8,
                bias=False
            ),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(self.drop_out)
        )

        self.block_3 = nn.Sequential(
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
        )
        self.self_attn = SelfAttention(in_channels=16)
        self.out = nn.Linear((16 * 246), classes_num)  # 修改全连接层的输入尺寸

    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.self_attn(x)
        x = x.view(x.size(0), -1)  # 将特征展平为一维向量
        x = self.out(x)  # 通过全连接层进行分类
        return x
    
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
