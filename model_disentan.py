import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from resnet import resnet50, resnet18
import torchvision
import copy
import torch.optim as optim

from model import count_parameters, Normalize, weights_init_kaiming, weights_init_classifier, ShallowModule, base_resnet

class Disentangle_net(nn.Module):
    def __init__(self, num_class, num_cam, arch1='resnet50', arch2='resnet18'):
        super(Disentangle_net, self).__init__()


        if arch1 == 'resnet50':
            self.pool_dim = 2048
            self.E2 = Encoder('resnet18', 2048)
        else:
            self.pool_dim = 512
            self.E2 = Encoder('resnet18', 512)
        self.E1 = Encoder(arch1, self.pool_dim)

        self.W1 = Classifier(self.pool_dim, num_class)
        self.W2 = Classifier(self.pool_dim, num_cam + 1)

        self.model_list = []
        self.model_list.append(self.E1)
        self.model_list.append(self.W1)
        self.model_list.append(self.E2)
        self.model_list.append(self.W2)

        self._init_optimizers()
        self.l2norm = Normalize(2)

    def forward(self, x1, x2, x3=None, modal=0):
        x = self.E1(x1, x2, x3=x3, modal=modal)
        x_a = self.g_avg(x)
        x_m = self.g_max(x)
        x_c = x_a + x_m
        if self.training:
            z = self.E2(x1, x2, x3=x3, modal=modal)
            z_a = self.g_avg(z)
            return x_a, x_m, x_c, z_a
        else:
            return self.l2norm(x_c), self.l2norm(self.W1.BN(x_c))

    def g_avg(self, x):
        x_pool = F.adaptive_avg_pool2d(x,1)
        return x_pool.view(x_pool.size(0), x_pool.size(1))

    def g_max(self, x):
        x_pool = F.adaptive_max_pool2d(x,1)
        return x_pool.view(x_pool.size(0), x_pool.size(1))

    def _init_optimizers(self):
        self.content_optimizer = optim.Adam(list(self.E1.parameters()), lr=0.00045, betas=(0.9, 0.999), weight_decay=5e-4)
        self.identifier_optimizer = optim.Adam(list(self.W1.parameters()), lr=0.00045, betas=(0.9, 0.999), weight_decay=5e-4)
        self.style_optimizer = optim.Adam(list(self.E2.parameters()), lr=0.00045, betas=(0.9, 0.999), weight_decay=5e-4)
        self.camClassifier_optimizer = optim.Adam(list(self.W2.parameters()), lr=0.00045, betas=(0.9, 0.999), weight_decay=5e-4)

        self.milestones = [181]
        self.content_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.content_optimizer, milestones=self.milestones, gamma=0.1)
        self.identifier_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.identifier_optimizer, milestones=self.milestones, gamma=0.1)
        self.style_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.style_optimizer, milestones=self.milestones, gamma=0.1)
        self.camClassifier_lr_scheduler = optim.lr_scheduler.MultiStepLR(self.camClassifier_optimizer, milestones=self.milestones, gamma=0.1)



    def lr_scheduler_step(self, current_epoch):
        self.content_lr_scheduler.step(current_epoch)
        self.identifier_lr_scheduler.step(current_epoch)
        self.style_lr_scheduler.step(current_epoch)
        self.camClassifier_lr_scheduler.step(current_epoch)


    def optimizers_zero(self):
        self.content_optimizer.zero_grad()
        self.identifier_optimizer.zero_grad()
        self.style_optimizer.zero_grad()
        self.camClassifier_optimizer.zero_grad()


    def optimizers_step(self):
        self.content_optimizer.step()
        self.identifier_optimizer.step()
        self.style_optimizer.step()
        self.camClassifier_optimizer.step()

    ## set model as train mode
    def set_train(self):
        self.train()
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii] = self.model_list[ii].train()

    ## set model as eval mode
    def set_eval(self):
        self.eval()
        for ii, _ in enumerate(self.model_list):
            self.model_list[ii] = self.model_list[ii].eval()

    def count_params(self):
        s = self.E1.count_params() + self.W1.count_params() + count_parameters(self.E2) + self.W2.count_params()
        return s

class Encoder(nn.Module):
    def __init__(self, arch='resnet50', feat_dim=2048):
        super(Encoder, self).__init__()

        self.thermal_module = ShallowModule(arch=arch)
        self.visible_module = ShallowModule(arch=arch)
        self.gray_module = ShallowModule(arch=arch)

        self.base_resnet = base_resnet(arch=arch)

        self.feat_dim = feat_dim
        if arch == 'resnet50':
            self.pool_dim = 2048
        else:
            if feat_dim == 2048:
                self.proj = nn.Conv2d(in_channels=512, out_channels=2048, kernel_size=1, stride=1,
                    padding=0)

            self.pool_dim = 512

        self.l2norm = Normalize(2)
        self.bottleneck = nn.BatchNorm1d(self.pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift

        self.bottleneck.apply(weights_init_kaiming)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        self.gm_pool = False

    def forward(self, x1, x2, x3=None, modal=0):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
            if x3 is not None :
                x3 = self.gray_module(x3)
                x = torch.cat((x, x3), 0)
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)
        elif modal == 3:
            x = self.gray_module(x3)

        # shared block

        x = self.base_resnet(x)
        if self.feat_dim != self.pool_dim:
            x = self.proj(x)
        # if self.bpool == 'gem':
        #     b, c, _, _ = x.shape
        #     x_pool = x.view(b, c, -1)
        #     p = 3.0
        #     x_pool = (torch.mean(x_pool**p, dim=-1) + 1e-12)**(1/p)
        # elif self.bpool == 'avg':
        #     x_pool = F.adaptive_avg_pool2d(x,1)
        #     x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        # elif self.bpool == 'max':
        #     x_pool = F.adaptive_max_pool2d(x,1)
        #     x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        # elif self.bpool == 'avg+max':
        #     x_pool_avg = F.adaptive_avg_pool2d(x,1)
        #     x_pool_avg = x_pool_avg.view(x_pool_avg.size(0), x_pool_avg.size(1))
        #
        #     x_pool_max = F.adaptive_max_pool2d(x,1)
        #     x_pool_max = x_pool_max.view(x_pool_max.size(0), x_pool_max.size(1))
        #     x_pool = x_pool_avg + x_pool_max
        return x
        if self.gm_pool  == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))
        feat = self.bottleneck(x_pool)

        retX_pool = x_pool
        retFeat = self.classifier(feat)

        if not self.training :
            retX_pool = self.l2norm(x_pool)
            retFeat = self.l2norm(feat)


        return retX_pool, retFeat

    def getPoolDim(self):
        return self.pool_dim

    def count_params(self):
        s = self.gray_module.count_params() * 2 + self.base_resnet.count_params()
        return s


class Classifier(nn.Module):
    def __init__(self, in_dim, num_class):
        super(Classifier, self).__init__()
        self.in_dim = in_dim
        self.num_class = num_class

        self.BN = nn.BatchNorm1d(self.in_dim)
        self.BN.bias.requires_grad_(False)

        self.classifier = nn.Linear(self.in_dim, self.num_class, bias=False)

        self.BN.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

    def forward(self, x):
        feat_afterBN = self.BN(x)

        cls_score = self.classifier(feat_afterBN)
        return cls_score, feat_afterBN

    def count_params(self):
        s = count_parameters(self.BN) + count_parameters(self.classifier)
        return s
