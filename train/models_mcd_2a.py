
import torch
import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from torch import nn
import torch.nn.functional as F
from torch import optim
import math

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

        # self.pos_encoder = PositionalEncoding(d_model=22)
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
        # 新增一个卷积块 block_4
        # self.block_4 = nn.Sequential(
        #     nn.Conv2d(
        #         in_channels=16,  # 输入通道数
        #         out_channels=32,  # 输出通道数
        #         kernel_size=(1, 8),  # 卷积核大小
        #         padding=(0, 3),  # 填充大小
        #         bias=False  # 是否使用偏置
        #     ),
        #     nn.BatchNorm2d(32),  # 批量归一化
        #     nn.ELU(),  # 激活函数
        #     nn.AvgPool2d((1, 4)),  # 平均池化
        #     nn.Dropout(self.drop_out)  # Dropout正则化
        # )
        # self.attn = MultiHeadAttention(d_model=16, num_heads=4)
        self.self_attn = SelfAttention(in_channels=16)
        self.out = nn.Linear((496), classes_num)

    def forward(self, x):
        x = self.block_1(x)
        # print("block1", x.shape)
        x = self.block_2(x)
        # print("block2", x.shape)
        x = self.block_3(x)
        # print("block3", x.shape)
        # x = self.block_4(x)
        x = self.self_attn(x)
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

# 注意力机制

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(2), :]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        self.v_linear = nn.Linear(d_model, d_model, bias=False)
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = nn.MultiheadAttention(d_model, num_heads)

    def forward(self, x):
        batch_size = x.size(0)
        v = self.v_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        attn_output, attn_output_weights = self.attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        return attn_output


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



# class SelfAttention(nn.Module):
#     def __init__(self, in_channels):
#         super(SelfAttention, self).__init__()
#         self.query = nn.Conv1d(in_channels, in_channels // 8, 1)
#         self.key = nn.Conv1d(in_channels, in_channels // 8, 1)
#         self.value = nn.Conv1d(in_channels, in_channels, 1)
#         self.gamma = nn.Parameter(torch.zeros(1))

#     def forward(self, x):
#         batch_size, channels, time = x.size()
        
#         # 计算 query, key, value
#         proj_query = self.query(x).view(batch_size, -1, time).permute(0, 2, 1)
#         proj_key = self.key(x).view(batch_size, -1, time)
#         proj_value = self.value(x).view(batch_size, -1, time)
        
#         # 计算注意力权重
#         energy = torch.bmm(proj_query, proj_key)
#         attention = torch.softmax(energy, dim=-1)
        
#         # 注意力加权
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(batch_size, channels, time)
        
#         # 残差连接
#         out = self.gamma * out + x
#         return out