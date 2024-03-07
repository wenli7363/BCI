
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import dataloader_2a
import models_mcd_2a
import numpy as np
# Training settings
# 先定义整个类别文件
def Entropy(input_):
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class Solver(object): 
    
    #取掉了args
    def __init__(self,  batch_size=128, number=2, learning_rate=0.001, interval=10,
                 # 优化器的类型
                 optimizer='adam'
                 , num_k=4):
        # 思考这个过程
        # 优先动k和学习率  k为10还是好一点 
        # 信号的尺寸也是可以调整的


        # 目前来说k为10的效果是比较好的，默认的k=4效果暂时没有10 好


        self.batch_size = batch_size
        self.num_k = num_k
        # self.checkpoint_dir = checkpoint_dir
        # self.save_epoch = save_epoch
        # self.use_abs_diff = args.use_abs_diff
        # self.all_use = all_use
        # if self.source == 'svhn':
        #     self.scale = True
        # else:
        #     self.scale = False
        print('dataset loading')

 ########################### # 加载训练和测试数据

        self.s_dataloaders,self.t_dataloaders = dataloader_2a.dataloader_2a(number=number)

        # number 作为目标域数据，剩下8个个体数据作为源域


        print('load finished!')

#########################################



        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
        ############定义模型

        self.G = models_mcd_2a.Feature(classes_num=4)
        self.C1 = models_mcd_2a.Predictor2(4)
        self.C2 = models_mcd_2a.Predictor2(4)




        ################

        self.G.to(self.device)
        self.C1.to(self.device)
        self.C2.to(self.device)
        self.interval = interval

        self.set_opimizer(which_opt=optimizer, lr=learning_rate)
        self.lr = learning_rate

# 2020cvpr 的一篇loss
#  改loss 
    def mcc_loss(self,input, class_num=4, temperature=2.5, train_bs=64):
        outputs_target_temp = input / temperature
        target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
        target_entropy_weight = Entropy(target_softmax_out_temp).detach()
        target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
        target_entropy_weight = train_bs * target_entropy_weight / torch.sum(target_entropy_weight)
        cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(
            target_softmax_out_temp)
        cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
        mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
        return mcc_loss


 

    ########## 定义优化器。##########
    def set_opimizer(self, which_opt='momentum', lr=0.001, momentum=0.9):
        if which_opt == 'momentum':
            self.opt_g = optim.SGD(self.G.parameters(),
                                   lr=lr, weight_decay=0.0005,
                                   momentum=momentum)

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

# coral 损失

    def CORAL(self, source, target):
        d = source.size(1)
        ns, nt = source.size(0), target.size(0)

        # source covariance
        tmp_s = torch.ones((1, ns)).to(self.device) @ source
        cs = (source.t() @ source - (tmp_s.t() @ tmp_s) / ns) / (ns - 1)

        # target covariance
        tmp_t = torch.ones((1, nt)).to(self.device) @ target
        ct = (target.t() @ target - (tmp_t.t() @ tmp_t) / nt) / (nt - 1)

        # frobenius norm
        loss = (cs - ct).pow(2).sum().sqrt()
        loss = loss / (4 * d * d)

        return loss

    def reset_grad(self):
        self.opt_g.zero_grad()
        self.opt_c1.zero_grad()
        self.opt_c2.zero_grad()

#   loss
    def ent(self, output):
        return - torch.mean(output * torch.log(output + 1e-6))


    # 算损失函数。
    def discrepancy(self, out1, out2):
        m = nn.Softmax(dim=1)
        return torch.mean(torch.abs(m(out1) - m(out2)))


# loss
    def cdd(self,output_t1, output_t2):
        mul = output_t1.transpose(0, 1).mm(output_t2)
        cdd_loss = torch.sum(mul) - torch.trace(mul)
        return cdd_loss

# loss
    def discrepancy1(self,out1, out2):
        m = nn.Softmax(dim=1)
        # (torch.pow((1 - probs), 2)
        # torch.mean() 放在一边

        a = torch.abs(m(out1) - m(out2))
        a1 = torch.ones_like(a)
        # return torch.mean(a * torch.pow((a - a1), 2))
        return torch.mean(a * a)

    def train(self, epoch, record_file=None):
        # 定义优化器
        #fl  = MultiCEFocalLoss(class_num=3).to(self.device)
        criterion = nn.CrossEntropyLoss().to(self.device)
        self.G.train()
        self.C1.train()
        self.C2.train()
        # 随机种子
        # torch.cuda.manual_seed(1)
        ######

        # 先都用test试一试

        data_zip = zip(self.s_dataloaders, self.t_dataloaders)

        for batch_idx, ((inputs_s, labels_s), (inputs_t, labels_t)) in enumerate(data_zip):

            inputs_s = inputs_s.to(self.device)
            inputs_t = inputs_t.to(self.device)
            labels_s = labels_s.to(self.device)
            labels_t = labels_t.to(self.device)

            self.reset_grad()

            # 这里额外定义了
            
### 这些都是直接分开定义好
            feat_s = self.G(inputs_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)

            ###### 1
            loss_s1 = criterion(output_s1, labels_s)
            loss_s2 = criterion(output_s2, labels_s)
            loss_s = loss_s1 + loss_s2
            loss_s.backward()
            self.opt_g.step()
            self.opt_c1.step()
            self.opt_c2.step()



###### 2  固定特征提取 用目标域训练分类器（最大化差异） 用原领域和目标与
            self.reset_grad()
            feat_s = self.G(inputs_s)
            output_s1 = self.C1(feat_s)
            output_s2 = self.C2(feat_s)
            feat_t = self.G(inputs_t)
            output_t1 = self.C1(feat_t)
            output_t2 = self.C2(feat_t)


            loss_s1 = criterion(output_s1, labels_s)
            loss_s2 = criterion(output_s2, labels_s)
            loss_s = loss_s1 + loss_s2
            loss_dis = self.discrepancy(output_t1, output_t2)
            loss = loss_s - loss_dis
            loss.backward()
            self.opt_c1.step()
            self.opt_c2.step()
            self.reset_grad()



######## 3  固定分类器，训练特征提取器。
            for i in range(self.num_k):
                #
                feat_t = self.G(inputs_t)
                output_t1 = self.C1(feat_t)
                output_t2 = self.C2(feat_t)
                # 0.5还可以
                # 这里mcc_loss系数要调一调。
                loss_dis = self.discrepancy(output_t1, output_t2)+0.5*self.mcc_loss(output_t1)
                loss_dis.backward()
                self.opt_g.step()
                self.reset_grad()

            if batch_idx % self.interval == 2:
                print('Train Epoch: {} [{}/{} ({:.4f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
                    epoch, batch_idx, 43,
                    100. * batch_idx / 43, loss_s1.item(), loss_s2.item(), loss_dis.item()))

        self.test(100)
        # 其实基于这种提升已经很高了。
        return batch_idx

    #
    #
    # def train_onestep(self, epoch, record_file=None):
    #     criterion = nn.MSELoss().cuda()
    #     self.G.train()
    #     self.C1.train()
    #     self.C2.train()
    #     torch.cuda.manual_seed(1)
    #     data_zip = zip(self.s_dataloaders['train'], self.t_dataloaders['train'])
    #     for batch_idx, ((inputs_s, labels_s), (inputs_t, labels_t)) in enumerate(data_zip):
    #
    #         inputs_s = inputs_s.cuda()
    #         inputs_t = inputs_t.cuda()
    #         labels_s=labels_s.cuda()
    #         labels_t=labels_t.cuda()
    #
    #         self.reset_grad()
    #         # print(inputs_s.shape)
    #         feat_s = self.G(inputs_s)
    #         output_s1 = self.C1(feat_s)
    #         output_s2 = self.C2(feat_s)
    #         loss_s1 = criterion(output_s1, labels_s)
    #         loss_s2 = criterion(output_s2, labels_s)
    #         loss_s = loss_s1 + loss_s2
    #
    #
    #         loss_s.backward()
    #         feat_t = self.G(inputs_t)
    #         self.C1.set_lambda(1.0)
    #         self.C2.set_lambda(1.0)
    #         output_t1 = self.C1(feat_t)
    #         output_t2 = self.C2(feat_t)
    #         loss_dis = -self.discrepancy(output_t1, output_t2)
    #
    #
    #         #loss_dis.backward()
    #         self.opt_c1.step()
    #         self.opt_c2.step()
    #         self.opt_g.step()
    #         self.reset_grad()
    #
    #         if batch_idx % self.interval == 0:
    #             print('Train Epoch: {} [{}/{} ({:.4f}%)]\tLoss1: {:.6f}\t Loss2: {:.6f}\t  Discrepancy: {:.6f}'.format(
    #                 epoch, batch_idx, 43,
    #                 100.* batch_idx/43, loss_s1.item(), loss_s2.item(), loss_dis.item()))
    #             # if record_file:
    #             #     record = open(record_file, 'a')
    #             #     record.write('%s %s %s\n' % (loss_dis.data[0], loss_s1.data[0], loss_s2.data[0]))
    #             #     record.close()
    #
    #     self.test(100)
    #     return batch_idx
    #
    #







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
           output_ensemble = output1 + output2
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

        # for batch_idx, data in enumerate(self.dataset_test):
        #     img = data['T']
        #     label = data['T_label']
        #     img, label = img.cuda(), label.long().cuda()
        #     img, label = Variable(img, volatile=True), Variable(label)
        #     feat = self.G(img)
        #     output1 = self.C1(feat)
        #     output2 = self.C2(feat)
        #     test_loss += F.nll_loss(output1, label).data[0]
        #     output_ensemble = output1 + output2
        #     pred1 = output1.data.max(1)[1]
        #     pred2 = output2.data.max(1)[1]
        #     pred_ensemble = output_ensemble.data.max(1)[1]
        #     k = label.data.size()[0]
        #     correct1 += pred1.eq(label.data).cpu().sum()
        #     correct2 += pred2.eq(label.data).cpu().sum()
        #     correct3 += pred_ensemble.eq(label.data).cpu().sum()
        #     size += k
        # test_loss = test_loss / size
        # print(
        #     '\nTest set: Average loss: {:.4f}, Accuracy C1: {}/{} ({:.0f}%) Accuracy C2: {}/{} ({:.0f}%) Accuracy Ensemble: {}/{} ({:.0f}%) \n'.format(
        #         test_loss, correct1, size,
        #         100. * correct1 / size, correct2, size, 100. * correct2 / size, correct3, size, 100. * correct3 / size))
        # if save_model and epoch % self.save_epoch == 0:
        #     torch.save(self.G,
        #                '%s/%s_to_%s_model_epoch%s_G.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
        #     torch.save(self.C1,
        #                '%s/%s_to_%s_model_epoch%s_C1.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
        #     torch.save(self.C2,
        #                '%s/%s_to_%s_model_epoch%s_C2.pt' % (self.checkpoint_dir, self.source, self.target, epoch))
        # if record_file:
        #     record = open(record_file, 'a')
        #     print('recording %s', record_file)
        #     record.write('%s %s %s\n' % (float(correct1) / size, float(correct2) / size, float(correct3) / size))
        #     record.close()



# from label_prop import label_prop



class MultiCEFocalLoss(torch.nn.Module):
    def __init__(self, class_num, gamma=2, alpha=None, reduction='mean'):
        super(MultiCEFocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    def forward(self, predict, target):

        pt = F.softmax(predict, dim=1) # softmmax获取预测概率
        class_mask = F.one_hot(target, 3) #获取target的one hot编码
        ids = target.view(-1, 1)
        alpha = self.alpha[ids.data.view(-1)].to(self.device) # 注意，这里的alpha是给定的一个list(tensor
#),里面的元素分别是每一个类的权重因子
        probs = (pt * class_mask).sum(1).view(-1, 1) # 利用onehot作为mask，提取对应的pt
        log_p = probs.log()
# 同样，原始ce上增加一个动态权重衰减因子
        loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()
        return loss


#用esayTL的数据

# 实例化
slover = Solver()
# 开始训练
for i in range(2000):
    slover.train(epoch=i)

# slover.test(100)

