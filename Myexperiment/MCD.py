import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from preprocessing import import_data
from dataloader import dataloader_train_val,dataloader_test
import model
import numpy as np

# Training settings
# 熵
def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class MCD_sovler(object): 
    
    #取掉了args
    # num_k表示训练几次特征提取器
    # number参数在这个代码中用于指定目标域的索引。（我自己的模型中似乎用不到）
    def __init__(self,  file_path,batch_size=128, number=3, learning_rate=0.001, interval=10,
                 # 优化器的类型
                 optimizer='adam',
                 num_k=4, alfa = 0.5,class_num=4):

        self.batch_size = batch_size
        self.alfa = alfa
        self.num_k = num_k
        self.best_ACC = 0    # 训练过程中，测试集最好的正确率
        self.number = number
        self.class_num = class_num

        self.file_path = file_path
        print('------------------------ dataset loading -------------------')

        
    #   ========================================== 加载训练集和测试集data_loader =======================================
        
        
        X, y = import_data(self.file_path)

        # 2. 创建数据加载器
        # 这里要改一下，改成分开加载源域和目标域的数据。
        self.s_dataloaders,self.t_dataloaders = dataloader_train_val(X, y, batch_size=batch_size, shuffle=True)

        print('------------------------ load finished!  ------------------------')

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        # 实例化模型
        self.G = model.Feature(classes_num=self.class_num)
        self.C1 = model.Predictor2(class_num)
        self.C2 = model.Predictor2(class_num)

        # 将模型转移到GPU上
        self.G.to(self.device)
        self.C1.to(self.device)
        self.C2.to(self.device)
        self.interval = interval

        self.set_opimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate

# 2020cvpr 的一篇loss
#  改loss 
# train_bs参数代表当前训练批次的大小(batch size)。
    def mcc_loss(self,input, class_num=4, temperature=2.5, train_bs=64):
        outputs_target_temp = input / temperature
        target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
        target_entropy_weight = Entropy(target_softmax_out_temp).detach()
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
        target_entropy_weight = train_bs * target_entropy_weight / torch.sum(target_entropy_weight)

        # cov_matrix_t计算目标域样本的加权协方差矩阵。
        cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(
            target_softmax_out_temp)
        cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
        # 计算协方差矩阵迹之外的其他元素之和,作为惩罚项。
        mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
        return mcc_loss


    ########## 定义优化器。##########
    def set_opimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum,)

            self.opt_c1 = optim.SGD(self.C1.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)
            self.opt_c2 = optim.SGD(self.C2.parameters(),
                                    lr=lr, weight_decay=0.0005,
                                    momentum=momentum)

        if which_opt == 'adam':
            self.opt_g = optim.Adam(self.G.parameters(),
                                    lr=lr, weight_decay=0.0005)

            self.opt_c1 = optim.Adam(self.C1.parameters(),
                                     lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.Adam(self.C2.parameters(),
                                     lr=lr, weight_decay=0.0005)
        if which_opt == 'adamw':
            self.opt_g = optim.AdamW(self.G.parameters(),
                                 lr=lr, weight_decay=0.0005)
            self.opt_c1 = optim.AdamW(self.C1.parameters(),
                                  lr=lr, weight_decay=0.0005)
            self.opt_c2 = optim.AdamW(self.C2.parameters(),
                                  lr=lr, weight_decay=0.0005)
            
    # 重置梯度
    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()

    # 交叉熵损失
    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))


    # 算损失函数。
    def discrepancy(self, out1, out2):
        m = nn.Softmax(dim=1)
        return torch.mean(torch.abs(m(out1) - m(out2)))


    def train(self, epoch, record_file=None):
        # 定义优化器
        criterion = nn.CrossEntropyLoss().to(self.device)
        self.G.train()
        self.C1.train()
        self.C2.train()
        

        data_zip = zip(self.s_dataloaders, self.t_dataloaders)
        for batch_idx, ((inputs_s, labels_s), (inputs_t, labels_t)) in enumerate(data_zip):

            # print(f"inputs_s shape: {inputs_s.shape}") 
            inputs_s = inputs_s.to(self.device) # inputs_s shape (batch_size,1,电极数,1000)
            inputs_t = inputs_t.to(self.device) 
            labels_s = labels_s.to(self.device) # (batch_size)
            labels_t = labels_t.to(self.device)

            self.reset_grad()
            # 打印输入尺寸
            # print(f"Input size: {inputs_s.size()}")
            
            """
            ************************************* 1 初始化分类器C1,C2,G ***************************************
            """
            feat_s = self.G(inputs_s)   # feat_s (batch_size, 992)
            # print(f"Feature extractor output size: {feat_s.size()}")

            output_s1 = self.C1(feat_s) #   (batch_size, 4) 得到4个类的预测概率
            # print(f"Classifier C1 output size: {output_s1.size()}")
            
            output_s2 = self.C2(feat_s)
            # print(f"Classifier C2 output size: {output_s2.size()}")

        
        #   直接用交叉熵损失来保证在源域上分类是准确的
            loss_s1 = criterion(output_s1, labels_s)
            loss_s2 = criterion(output_s2, labels_s)
            loss_s = loss_s1 + loss_s2
            loss_s.backward()
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()

            """
            ***************************** 2 固定特征提取 用目标域训练分类器（最大化差异） 用原域和目标域 ***************************
            """

            # 训练K轮F
            for i in range(self.num_k):
                # 源域
                feat_s = self.G(inputs_s)
                output_s1 = self.C1(feat_s)
                output_s2 = self.C2(feat_s)
                # 目标域
                feat_t = self.G(inputs_t)
                output_t1 = self.C1(feat_t)     # (batch_size, 4)
                output_t2 = self.C2(feat_t)

                # 计算源域上的交叉熵损失
                loss_s1 = criterion(output_s1, labels_s)        
                loss_s2 = criterion(output_s2, labels_s)
                loss_s = loss_s1 + loss_s2
                loss_dis = self.discrepancy(output_t1, output_t2)           # 目标域上的差异
                
                # 分类器各自的交叉熵损失-分类结果差异
                # 希望最大化各自的分类能力，同时最大化分类差异
                # 结果是源域上的分类能力更强，同时考虑了目标域上特征提取能力。
                loss = loss_s - loss_dis
                loss.backward()
                self.opt_c1.step()
                self.opt_c2.step()
                self.reset_grad()



            """
            *************************************************** 固定分类器，训练特征提取器。 **********************************************
            """
            for i in range(self.num_k):
                #
                feat_t = self.G(inputs_t)   # 输入数据被映射为(batch_size, 992)的特征向量。

                # 分类器输入维度为(batch_size, 992),输出维度为(batch_size, 4),对应4个类别的预测概率。
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                # 0.5还可以
                # 这里mcc_loss系数要调一调。
                # 两分类器的输出分布差异要小，同时最大化不同类别之间预测概率的差异性。
                loss_dis = self.discrepancy(output_t1, output_t2)+self.alfa*self.mcc_loss(output_t1)
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()

            # 每10个batch打印一次
            if batch_idx % self.interval == 2:
                print('Train Epoch: {} [{}/{} ({:.4f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                    epoch, batch_idx, 43,
                    100. * batch_idx / 43, loss_s1.item(), loss_s2.item(), loss_dis.item()))

        # 测试100次
        tmpacc = self.test(100)
        if tmpacc > self.best_ACC : 
            self.best_ACC = tmpacc
        # 其实基于这种提升已经很高了。
        return batch_idx



    def test(self, epoch, record_file=None, save_model=False):

        self.G.eval()
        self.C1.eval()
        self.C2.eval()

        test_loss = 0
        correct1 = 0
        correct2 = 0
        correct3 = 0
        size = 0
        # 调整test 和train 的使用区间
        data_zip = zip(self.s_dataloaders, self.t_dataloaders)
        for batch_idx, (( _ , _ ), (inputs_t, labels_t)) in enumerate(data_zip):
           inputs_t=inputs_t.to(self.device)
           labels_t=labels_t.to(self.device)
           feat = self.G(inputs_t)
           output1 = self.C1(feat)
           output2 = self.C2(feat)
           test_loss += F.nll_loss(output1, labels_t).data
           output_ensemble = output1 + output2              # 集成分类器结果
           pred1 = output1.data.max(1)[1]
           pred2 = output2.data.max(1)[1]
           pred_ensemble = output_ensemble.data.max(1)[1]
           k = labels_t.data.size()[0]
           correct1 += pred1.eq(labels_t.data).cpu().sum()
           correct2 += pred2.eq(labels_t.data).cpu().sum()
           correct3 += pred_ensemble.eq(labels_t.data).cpu().sum()
           size += k
        test_loss = test_loss / size

        print(
            '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(
                test_loss, correct1, size,
                100. * correct1 / size, correct2, size, 100. * correct2 / size, correct3, size, 100. * correct3 / size))
       
        return 100. * correct3 / size

"""
 ============================================== 开始跑模型的地方 =========================================
"""


filePath = 'D:\\Desktop\\4\\42.h5'
# 实例化
slover = MCD_sovler(file_path=filePath,batch_size = 128,learning_rate = 0.005,class_num = 4)
   
# 开始训练
start = time.time() 
# 训练1000次
for i in range(2):
    slover.train(epoch=i)
end = time.time()
execution_time = end - start