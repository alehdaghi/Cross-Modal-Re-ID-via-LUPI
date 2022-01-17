import torch
import torch.nn as nn
from einops import rearrange
from torch.nn import init
from torch.nn import functional as F

from model import ShallowModule, make_conv_params_same, base_resnet,\
    Non_local, Normalize, weights_init_kaiming,weights_init_classifier
from resnet import resnet50, resnet18
import torchvision
import copy

from sup_con_loss import SupConLoss


class embed_net(nn.Module):
    def __init__(self,  class_num, no_local= 'on', gm_pool = 'on', arch='resnet50'):
        super(embed_net, self).__init__()
        if arch == 'resnet50':
            self.pool_dim = 2048
        else:
            self.pool_dim = 512

        self.thermal_module = ShallowModule(arch=arch)
        self.visible_module = ShallowModule(arch=arch)
        self.gray_module = ShallowModule(arch=arch)

        self.base_resnet = base_resnet(arch=arch)
        self.partFeat_module = copy.deepcopy(self.base_resnet.resnet_part2[2]) # layer4

        self.dim = 0
        self.part_num = 5

        spatial_attention = nn.Conv2d(self.pool_dim, self.part_num, kernel_size=1, stride=1, padding=0, bias=True)
        torch.nn.init.constant_(spatial_attention.bias, 0.0)
        activation = nn.Sigmoid()
        self.maskDetector = nn.Sequential(spatial_attention, activation)

        self.partsWeight = nn.Sequential(nn.Linear(self.pool_dim, 1), nn.GELU())

        self.non_local = no_local
        if self.non_local =='on':
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



        self.l2norm = Normalize(2)

        self.bottleneck = nn.BatchNorm1d(2 * self.pool_dim)
        self.bottleneck.bias.requires_grad_(False)  # no shift
        nn.init.constant_(self.bottleneck.bias, 0)
        self.classifier = nn.Linear(2 * self.pool_dim, class_num, bias=False)
        self.bottleneck.apply(weights_init_kaiming)
        self.classifier.apply(weights_init_classifier)

        # self.bottleneck_parts = [nn.BatchNorm1d(self.pool_dim) for i in range(self.part_num)]
        # for btl in self.bottleneck_parts:
        #     btl.bias.requires_grad_(False)  # no shift
        #     nn.init.constant_(btl.bias, 0)
        #     btl.apply(weights_init_kaiming)
        #
        # self.classifier_parts = [nn.Linear(self.pool_dim, class_num, bias=False) for i in range(self.part_num)]
        # for cls in self.classifier_parts:
        #     cls.apply(weights_init_classifier)
        self.criterion_contrastive = SupConLoss()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gm_pool = gm_pool




    def forward(self, x1, x2, x3=None, modal=0, with_feature = False):
        if modal == 0:
            x1 = self.visible_module(x1)
            x2 = self.thermal_module(x2)
            x = torch.cat((x1, x2), 0)
            view_size = 2
            if x3 is not None :
                x3 = self.gray_module(x3)
                x = torch.cat((x, x3), 0)
                view_size = 3
        elif modal == 1:
            x = self.visible_module(x1)
        elif modal == 2:
            x = self.thermal_module(x2)
        elif modal == 3:
            x = self.gray_module(x3)

        # shared block
        if self.non_local == 'on':
            NL1_counter = 0
            if len(self.NL_1_idx) == 0: self.NL_1_idx = [-1]
            # for i in range(len(self.base_resnet.base.layer1)):
            #     x = self.base_resnet.layer1[i](x)
            #     if i == self.NL_1_idx[NL1_counter]:
            #         _, C, H, W = x.shape
            #         x = self.NL_1[NL1_counter](x)
            #         NL1_counter += 1
            # Layer 2
            NL2_counter = 0
            if len(self.NL_2_idx) == 0: self.NL_2_idx = [-1]
            for i in range(len(self.base_resnet.layer2)):
                x = self.base_resnet.layer2[i](x)
                if i == self.NL_2_idx[NL2_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_2[NL2_counter](x)
                    NL2_counter += 1
            # Layer 3
            NL3_counter = 0
            if len(self.NL_3_idx) == 0: self.NL_3_idx = [-1]
            for i in range(len(self.base_resnet.layer3)):
                x = self.base_resnet.layer3[i](x)
                if i == self.NL_3_idx[NL3_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_3[NL3_counter](x)
                    NL3_counter += 1
            # Layer 4
            bodyFeat = self.partFeat_module(x)
            NL4_counter = 0
            if len(self.NL_4_idx) == 0: self.NL_4_idx = [-1]
            for i in range(len(self.base_resnet.layer4)):
                x = self.base_resnet.layer4[i](x)
                if i == self.NL_4_idx[NL4_counter]:
                    _, C, H, W = x.shape
                    x = self.NL_4[NL4_counter](x)
                    NL4_counter += 1
        else:
            x = self.base_resnet.resnet_part2[0](x) # layer2
            x = self.base_resnet.resnet_part2[1](x) # layer3
            if self.training:
                bodyFeat = self.partFeat_module(x)
            x = self.base_resnet.resnet_part2[2](x) # layer4

        masks = self.extractPartsMask(x)
        parts_feat = [self.gl_pool(x*m) for m in masks.split(1, dim=1)]
        parts_weight = torch.cat([self.partsWeight(part) for part in parts_feat], dim=-1).softmax(dim=1)
        #parts_score = [self.classifier_parts[i](self.bottleneck_parts[i](pf))
        #               for (i, pf) in enumerate(parts_feat)]
        parts_feat = torch.stack(parts_feat, dim = 1)
        weighted_part_feat = torch.einsum('b p, b p c -> b c', parts_weight, parts_feat)



        loss_body_cont, loss_cont, loss_mask = 0, 0, 0
        if self.training:
            bodyFeatParts = torch.stack([self.gl_pool(bodyFeat*m) for m in masks.split(1, dim=1)], dim=0) # index of each part matters
            b, _, w, h = x.shape
            masks_correlation = torch.einsum('b p w h, b c w h -> b p c', masks , masks)
            loss_mask = torch.triu(masks_correlation, diagonal = 1).sum() / (b * self.part_num * (self.part_num - 1) / 2)
            diag = torch.diagonal(masks_correlation, dim1=1, dim2=2)
            for i in range(self.part_num):
                loss_reg_mask = w * h / (2*self.part_num) - diag[:,i].sum() / b
                if loss_reg_mask.item() > 0 :
                    loss_mask =  loss_mask + loss_reg_mask
            # p = rearrange(F.normalize(bodyFeatParts, dim=1), '(v b p) ... -> b (v p) ...', v = view_size, b=b // view_size)
            loss_body_cont = self.criterion_contrastive(F.normalize(bodyFeatParts, dim=2))


        x_global = self.gl_pool(x)
        feat_pool = torch.cat([x_global, weighted_part_feat], dim=1)
        feat = self.bottleneck(feat_pool)

        if with_feature:
            return feat_pool, feat, masks, x, parts_weight

        if not self.training :
            return self.l2norm(feat), self.l2norm(feat_pool)

        partedMap = {'feats': weighted_part_feat,
                     # 'score': parts_score,
                     'loss_mask': loss_mask,
                     'loss_body_cont': loss_body_cont,
                     }

        return feat_pool, self.classifier(feat), partedMap


    def gl_pool(self, x):
        if self.gm_pool  == 'on':
            b, c, h, w = x.shape
            x = x.view(b, c, -1)
            p = 3.0
            x_pool = (torch.mean(x**p, dim=-1) + 1e-12)**(1/p)
        else:
            x_pool = self.avgpool(x)
            x_pool = x_pool.view(x_pool.size(0), x_pool.size(1))

        return x_pool

    def extractPartsMask(self, x):
        return self.maskDetector(x)

    def getPoolDim(self):
        return 2*self.pool_dim



    def count_params(self):
        ids = set(map(id, self.parameters()))
        params = filter(lambda p: id(p) in ids, self.parameters())
        return sum(p.numel() for p in params if p.requires_grad)

