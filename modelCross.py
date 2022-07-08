import torch
import torch.nn as nn
from torch.nn import init
from resnet import resnet50, resnet18
import torchvision
import copy
import torch.nn.functional as F

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm)
        return out

class Non_local(nn.Module):
    def __init__(self, in_channels, reduc_ratio=4):
        super(Non_local, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = reduc_ratio//reduc_ratio

        self.g = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1, stride=1,
                    padding=0),
        )

        self.W = nn.Sequential(
            nn.Conv2d(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.in_channels),
        )
        nn.init.constant_(self.W[1].weight, 0.0)
        nn.init.constant_(self.W[1].bias, 0.0)
        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)

        self.phi = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        '''
                :param x: (b, c, t, h, w)
                :return:
                '''

        batch_size = x.size(0)
        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)
        N = f.size(-1)
        # f_div_C = torch.nn.functional.softmax(f, dim=-1)
        f_div_C = f / N

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z




class ShallowModule(nn.Module):
    def __init__(self, arch='resnet50'):
        super(ShallowModule, self).__init__()
        resnet = torchvision.models.resnet50(pretrained=True)
        # avg pooling to global pooling
        self.resnet_part1 = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.maxpool,  # no relu
            resnet.layer1)

    def forward(self, x):
        x = self.resnet_part1(x)
        return x


class base_resnet(nn.Module):
    def __init__(self):
        super(base_resnet, self).__init__()

        resnet = torchvision.models.resnet50(pretrained=True)
        resnet.layer4[0].conv2.stride = (1, 1)
        # avg pooling to global pooling
        resnet.layer4[0].downsample[0].stride = (1, 1)
        self.resnet_part2 = nn.Sequential(resnet.layer2, resnet.layer3, resnet.layer4)

    def forward(self, x):
        x = self.resnet_part2(x)
        return x


class embed_net(nn.Module):
    def __init__(self,  class_num=395):
        super(embed_net, self).__init__()

        self.thermal_module = ShallowModule()
        self.visible_module = ShallowModule()
        self.gray_module = ShallowModule()


        self.base_resnet = base_resnet()


        layers=[3, 4, 6, 3]
        non_layers=[0,2,3,0]
        self.NL_1 = nn.ModuleList(
            [Non_local(256) for i in range(non_layers[0])])
        self.NL_1_idx = sorted([layers[0] - (i + 1) for i in range(non_layers[0])])
        self.NL_2 = nn.ModuleList(
            [Non_local(512) for i in range(non_layers[1])])
        self.NL_2_idx = sorted([layers[1] - (i + 1) for i in range(non_layers[1])])
        self.NL_3 = nn.ModuleList(
            [Non_local(1024) for i in range(non_layers[2])])
        self.NL_3_idx = sorted([layers[2] - (i + 1) for i in range(non_layers[2])])
        self.NL_4 = nn.ModuleList(
            [Non_local(2048) for i in range(non_layers[3])])
        self.NL_4_idx = sorted([layers[3] - (i + 1) for i in range(non_layers[3])])

        self.pool_dim = 2048

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(self.pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        self.classifier = nn.Linear(self.pool_dim, class_num, bias=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x1, x2, x3=None, modal=0):
        if modal == 0: # train
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
            if x3 is not None :
                x3 = self.gray_module(x3)
                x = torch.cat((x, x3), 0)
        elif modal == 1: # RGB
            x = self.visible_module(x1)
        elif modal == 2: # IR
            x = self.thermal_module(x2)
        elif modal == 3: # random channels of RGB (a*R, b*G, c*B) s.o a+b+c=1 and a,b in rand(0,1)
            x = self.gray_module(x3)

        # shared block
        NL2_counter = 0
        if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
        for i in range(len(self.base_resnet.resnet_part2[0])):
            x = self.base_resnet.resnet_part2[0][i](x)
            if i == self.NL_2_idx[NL2_counter]:
                x = self.NL_2[NL2_counter](x)
                NL2_counter += 1
        # Layer 3
        NL3_counter = 0
        if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
        for i in range(len(self.base_resnet.resnet_part2[1])):
            x = self.base_resnet.resnet_part2[1][i](x)
            if i == self.NL_3_idx[NL3_counter]:
                x = self.NL_3[NL3_counter](x)
                NL3_counter += 1
        # Layer 4
        NL4_counter = 0
        if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
        for i in range(len(self.base_resnet.resnet_part2[2])):
            x = self.base_resnet.resnet_part2[2][i](x)
            if i == self.NL_4_idx[NL4_counter]:
                x = self.NL_4[NL4_counter](x)
                NL4_counter += 1

        b, c, h, w = x.shape
        x = x.view(b, c, -1)
        p = 3.0
        x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)

        feat = self.bottleneck(x_pool)

        if self.training :
            return x_pool, self.classifier(feat)
        else:
            return self.l2norm(x_pool), self.l2norm(feat)

    def getPoolDim(self):
        return self.pool_dim

    def count_params(self):
        ids = set(map(id, self.parameters()))
        params = filter(lambda p: id(p) in ids, self.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

