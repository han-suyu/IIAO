
import torch.nn as nn
import torch
from torch.utils import model_zoo
import numpy as np
import torch.nn.functional as F

np.set_printoptions(threshold=np.inf) 
np.set_printoptions(suppress=True)
################################################################################
# IIAO
################################################################################
class IIAO(nn.Module):
    def __init__(self):
        super(IIAO, self).__init__()
        self.vgg = VGG()
        self._load_vgg()
        self.dmp = BackEnd()

        

    def forward(self, input):
        input = self.vgg(input)
        dmp_out = self.dmp(*input)

        return dmp_out

    def _load_vgg(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg16_bn-6c64b313.pth')
        old_name = [0, 1, 3, 4, 7, 8, 10, 11, 14, 15, 17, 18, 20, 21, 24, 25, 27, 28, 30, 31, 34, 35, 37, 38, 40, 41]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '4_1', '4_2', '4_3', '5_1', '5_2', '5_3']
        new_dict = {}
        for i in range(13):
            new_dict['conv' + new_name[i] + '.conv.weight'] = state_dict['features.' + str(old_name[2 * i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = state_dict['features.' + str(old_name[2 * i]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.weight'] = state_dict['features.' + str(old_name[2 * i + 1]) + '.weight']
            new_dict['conv' + new_name[i] + '.bn.bias'] = state_dict['features.' + str(old_name[2 * i + 1]) + '.bias']
            new_dict['conv' + new_name[i] + '.bn.running_mean'] = state_dict['features.' + str(old_name[2 * i + 1]) + '.running_mean']
            new_dict['conv' + new_name[i] + '.bn.running_var'] = state_dict['features.' + str(old_name[2 * i + 1]) + '.running_var']

        self.vgg.load_state_dict(new_dict)

        


class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()

        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = BaseConv(1024, 512, 1, 1, activation=nn.ReLU(), use_bn=True)
        self.final_conv = BaseConv(512, 1, 1, 1, activation=nn.ReLU(), use_bn=False)

        self.fpm_a = ASP(512)
        self.gam_a = TAU(512)

        self.fpm_b = ASP(512)
        self.gam_b = TAU(512)

       

    def forward(self, *x):
        conv4_3, conv5_3 = x

        
        conv5_3 = self.upsample(conv5_3)
        input = self.conv(torch.cat([conv5_3, conv4_3], 1))
       

        fpm_a = self.fpm_a(input)
        gam_a = self.gam_a(input)  
       
      
        soft_a = torch.nn.functional.softmax(gam_a, 1)
        map_a = torch.sum(fpm_a*soft_a,1, keepdim=True)
        fpm_a = fpm_a*gam_a
      
    

        fpm_b = self.fpm_b(fpm_a)
        gam_b = self.gam_b(fpm_a)

        soft_b = torch.nn.functional.softmax(gam_b, 1)
        map_b = torch.sum(fpm_b*soft_b,1, keepdim=True)
        fpm_b = fpm_b*gam_b

    
     
        pr_density = self.final_conv(fpm_b)



        return map_a,map_b,pr_density




class TAU(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(TAU, self).__init__()

        self.conv1 = BaseConv(in_planes, round(in_planes // ratio), 1, 1, activation=nn.ReLU(), use_bn=False)
        # self.conv2 = BaseConv(round(in_planes // ratio), round(in_planes // ratio), 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv3 = BaseConv(round(in_planes // ratio), in_planes, 1, 1, activation=nn.Sigmoid(), use_bn=False)

    def forward(self, input):
        out = self.conv1(input)
        # out = self.conv2(out)
        out = self.conv3(out)

        return out


class ASP(nn.Module):
    def __init__(self, in_channels):
        super(ASP, self).__init__()

        self.branch_1 = nn.Sequential(
            BaseConv(in_channels, in_channels//4,    3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(in_channels//4, in_channels//4, 1, 1,activation=nn.ReLU(), use_bn=True)
            )

        self.branch_2 = nn.Sequential(
            BaseConv(in_channels, in_channels//4,    3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(in_channels//4, in_channels//4, 3, 1,activation=nn.ReLU(), use_bn=True)
            )

        self.branch_3 = nn.Sequential(
            BaseConv(in_channels, in_channels//4,    3, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(in_channels//4, in_channels//4, 5, 1,activation=nn.ReLU(), use_bn=True)
            )

        self.branch_4 = nn.Sequential(
            BaseConv(in_channels, in_channels//4,    5, 1, activation=nn.ReLU(), use_bn=True),
            BaseConv(in_channels//4, in_channels//4, 5, 1,activation=nn.ReLU(), use_bn=True)
            )


    def forward(self, input):
        branch_1 = self.branch_1(input)
        branch_2 = self.branch_2(input)
        branch_3 = self.branch_3(input)
        branch_4 = self.branch_4(input)
      
        out = torch.cat((branch_1, branch_2, branch_3, branch_4), dim=1)

        return out 
    
    
    
#This part of the code is borrowed from https://github.com/pxq0312/SFANet-crowd-counting
class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(3, 64, 3, 1,activation=nn.ReLU(), use_bn=True)
        self.conv1_2 = BaseConv(64, 64, 3, 1,activation=nn.ReLU(), use_bn=True)
        self.conv2_1 = BaseConv(64, 128, 3, 1,activation=nn.ReLU(), use_bn=True)
        self.conv2_2 = BaseConv(128, 128, 3, 1,activation=nn.ReLU(), use_bn=True)
        self.conv3_1 = BaseConv(128, 256, 3, 1,activation=nn.ReLU(), use_bn=True)
        self.conv3_2 = BaseConv(256, 256, 3, 1,activation=nn.ReLU(), use_bn=True)
        self.conv3_3 = BaseConv(256, 256, 3, 1,activation=nn.ReLU(), use_bn=True)
        self.conv4_1 = BaseConv(256, 512, 3, 1,activation=nn.ReLU(), use_bn=True)
        self.conv4_2 = BaseConv(512, 512, 3, 1,activation=nn.ReLU(), use_bn=True)
        self.conv4_3 = BaseConv(512, 512, 3, 1,activation=nn.ReLU(), use_bn=True)
        self.conv5_1 = BaseConv(512, 512, 3, 1,activation=nn.ReLU(), use_bn=True)
        self.conv5_2 = BaseConv(512, 512, 3, 1,activation=nn.ReLU(), use_bn=True)
        self.conv5_3 = BaseConv(512, 512, 3, 1,activation=nn.ReLU(), use_bn=True)


    def forward(self, input):
        input = self.conv1_1(input)
        conv1_2 = self.conv1_2(input)

        input = self.pool(conv1_2)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        conv3_3 = self.conv3_3(input)

        input = self.pool(conv3_3)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        conv4_3 = self.conv4_3(input)

        input = self.pool(conv4_3)
        input = self.conv5_1(input)
        input = self.conv5_2(input)
        conv5_3 = self.conv5_3(input)

        return conv4_3, conv5_3


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False):
        super(BaseConv, self).__init__()
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, kernel//2)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


if __name__=='__main__':
    input = torch.randn(4,3,400,400)
    model = IIAO()
    map_a,map_b, pr_density = model(input)
    print(map_a.shape, map_b.shape,  pr_density.shape)
