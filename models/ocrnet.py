import torch
import torch.nn as nn
import torch.nn.functional as F
from models.submodule import *

def label_to_onehot(gt, num_classes):
    '''
    :param gt: groundtruth with size (N,H,W)
    :param num_classes: the number of classes of different label
    return: (N, K, H, W)
    '''
    N, H, W = gt.size()
    onehot = torch.zeros(N, H, W, num_classes).cuda()
    onehot = onehot.scatter_(-1, gt.unsqueeze(-1), 1)

    return onehot.permute(0, 3, 1, 2)

class ObjectAttention(nn.Module):
    def __init__(self, in_channels, key_channels, out_channels, use_gt=False):
        super(ObjectAttention,self).__init__()
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.out_channels = out_channels
        self.use_gt = use_gt
        self.phi = nn.Sequential(convbn(self.in_channels, self.key_channels, 1, 1, 0, 1),
                                 nn.ReLU(),
                                 convbn(self.key_channels, self.key_channels, 1, 1, 0, 1),
                                 nn.ReLU())
        self.psi = nn.Sequential(convbn(self.in_channels, self.key_channels, 1, 1, 0, 1),
                                 nn.ReLU(),
                                 convbn(self.key_channels, self.key_channels, 1, 1, 0, 1),
                                 nn.ReLU())
        self.f_down = nn.Sequential(convbn(self.in_channels, self.key_channels, 1, 1, 0, 1),
                                 nn.ReLU())
        self.f_up = nn.Sequential(convbn(self.key_channels, self.in_channels, 1, 1, 0, 1),
                                 nn.ReLU())

    def forward(self, x, f_k, disp_gt=None):
        batch_size, _, h, w = x.shape
        # x:(N,C,H,W), f_k:(N,C,K,1)
        query = self.psi(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1) # N * hw * c
        key = self.phi(f_k).view(batch_size, self.key_channels, -1)
        value = self.f_down(f_k).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1) # N * k * c

        # 3. 计算权重w
        if self.use_gt and disp_gt is not None:
            gt_labels = label_to_onehot(disp_gt.squeeze(1).type(torch.cuda.LongTensor), 48)
            sim_map = gt_labels.permute(0, 2, 3, 1).view(batch_size, h*w, -1)
            sim_map = F.normalize(sim_map, p=1, dim=-1)
        else:
            sim_map = torch.matmul(query, key) # N * hw * k
            sim_map = F.softmax(sim_map, dim=-1) # 在k维上做的相似性度量
        # 4. Object contextual representations
        context = torch.matmul(sim_map, value) # N * hw * c
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context) # N * c * h * w

        return context

class OCR(nn.Module):
    def __init__(self, d_num, c_num, dropout=0.05, use_gt=False):
        super(OCR, self).__init__()
        self.d_num = d_num
        self.c_num = c_num
        self.dropout = dropout
        self.use_gt = use_gt
        self.objAtten = ObjectAttention(in_channels=256, key_channels=128, out_channels=256, use_gt=self.use_gt)
        self.disp_conv = nn.Sequential(convbn(self.d_num*self.c_num, 256, 1, 1, 0, 1),
                      nn.ReLU())
        self.final_conv = nn.Sequential(convbn(512, 256, 1, 1, 0, 1),
                               nn.ReLU(),
                               nn.Dropout2d(self.dropout))
        self.classifier = nn.Conv2d(256, self.d_num, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x, cost, disp_gt=None):
        if self.use_gt and disp_gt is not None:
            disp_gt = F.interpolate(disp_gt, scale_factor=0.25) // 4
            if disp_gt.max()>47:
                print("disp_gt min:{}, max:{}".format(disp_gt.min(),disp_gt.max()))
            gt_probs = label_to_onehot(disp_gt.squeeze(1).type(torch.cuda.LongTensor), 48)
            B, c_num, n_class, H, W = x.shape
            gt_probs = gt_probs.view(B, n_class, -1)
            M = F.normalize(gt_probs, p=1, dim=2)# batch x k x hw
        else:
            cost = torch.squeeze(cost, 1) #[1, 48, 64, 128]
            cost = F.softmax(cost, dim=1)
            # 1. d张概率图（这里d=D/4=48）
            B, c_num, n_class, H, W = x.shape
            d_flat = cost.view(B, n_class, -1)
            # M:(N,K,hw)
            M = F.normalize(d_flat, p=1, dim=2)

        # 2. 计算每个d类特征
        # 2.1 消除d维度
        x_new = x.view(B, c_num*n_class, H, W)
        feats = self.disp_conv(x_new)
        feats = torch.squeeze(feats, 1) # (N,C,H,W)
        channel = feats.shape[1]
        x_flat = feats.view(B, channel, -1)  # x:(N,C,L)
        x_flat = x_flat.permute(0, 2, 1)  # x:(N,L,C)
        d_feas = torch.matmul(M, x_flat).permute(0, 2, 1).unsqueeze(3)  # (N,C,K,1)

        y = self.objAtten(feats, d_feas, disp_gt) # y:(N,C,H,W)
        output = self.final_conv(torch.cat([y, feats], 1))
        output = self.classifier(output)
        return output, M.view(B, n_class, H, W)

