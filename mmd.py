from __future__ import absolute_import
from __future__ import division

import torch
import torch.nn as nn

import numpy as np  #
from scipy.spatial import distance  #
from scipy.stats import norm  #
import matplotlib.pyplot as plt  #
import seaborn as sns  #
import pickle  #
import torch  #
from sklearn.cluster import KMeans #
import random
from torchreid.metrics import compute_distance_matrix

from functools import partial
from torch.autograd import Variable


class MaximumMeanDiscrepancy(nn.Module):

    """
    Implementation of MMD :
    https://github.com/shafiqulislamsumon/HARTransferLearning/blob/master/maximum_mean_discrepancy.py
    """

    def __init__(self, use_gpu=True, batch_size=32, instances=4, global_only=False, distance_only=True):
        super(MaximumMeanDiscrepancy, self).__init__()
        self.use_gpu = use_gpu
        self.batch_size = batch_size
        self.instances = instances
        self.global_only = global_only
        self.distance_only = distance_only

    # Consider linear time MMD with a linear kernel:
    # K(f(x), f(y)) = f(x)^Tf(y)
    # h(z_i, z_j) = k(x_i, x_j) + k(y_i, y_j) - k(x_i, y_j) - k(x_j, y_i)
    #             = [f(x_i) - f(y_i)]^T[f(x_j) - f(y_j)]
    #
    # f_of_X: batch_size * k
    # f_of_Y: batch_size * k
    def mmd_linear(self, f_of_X, f_of_Y):
        delta = f_of_X - f_of_Y
        loss = torch.mean(torch.mm(delta, torch.transpose(delta, 0, 1)))
        return loss

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)#/len(kernel_val)

    def mmd_rbf_accelerate(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target,
            kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        loss = 0
        for i in range(batch_size):
            s1, s2 = i, (i+1)%batch_size
            t1, t2 = s1+batch_size, s2+batch_size
            loss += kernels[s1, s2] + kernels[t1, t2]
            loss -= kernels[s1, t2] + kernels[s2, t1]
        return loss / float(batch_size)

    def mmd_rbf_noaccelerate(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target,
                                  kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

    def pairwise_distance(self, x, y):

        if not len(x.shape) == len(y.shape) == 2:
            raise ValueError('Both inputs should be matrices.')

        if x.shape[1] != y.shape[1]:
            raise ValueError('The number of features should be the same.')

        x = x.view(x.shape[0], x.shape[1], 1)
        y = torch.transpose(y, 0, 1)
        output = torch.sum((x - y) ** 2, 1)
        output = torch.transpose(output, 0, 1)
        return output

    def gaussian_kernel_matrix(self, x, y, sigmas):
        sigmas = sigmas.view(sigmas.shape[0], 1)
        beta = 1. / (2. * sigmas)
        dist = self.pairwise_distance(x, y).contiguous()
        dist_ = dist.view(1, -1)
        s = torch.matmul(beta, dist_.cuda())
        return torch.sum(torch.exp(-s), 0).view_as(dist)

    def maximum_mean_discrepancy(self, x, y, kernel=gaussian_kernel_matrix):
        cost = torch.mean(kernel(x, x))
        cost += torch.mean(kernel(y, y))
        cost -= 2 * torch.mean(kernel(x, y))
        return cost

    def mmd_loss(self, source, target):

        sigmas = [
            1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
            1e3, 1e4, 1e5, 1e6
        ]
        gaussian_kernel = partial(
                self.gaussian_kernel_matrix, sigmas=Variable(torch.cuda.FloatTensor(sigmas))
            )
        loss_value = self.maximum_mean_discrepancy(source, target, kernel=gaussian_kernel)
        loss_value = loss_value
        return loss_value

    def forward(self, source_features, target_features):

        # group each images of the same identity together
        instances = self.instances
        batch_size = self.batch_size
        feature_size = target_features.shape[1]
        t = torch.reshape(target_features, (int(batch_size / instances), instances, feature_size))

        if self.global_only:
            return self.mmd_loss(source_features, target_features)


            #  and compute bc/wc euclidean distance
        wct = compute_distance_matrix(t[0], t[0])
        bct = compute_distance_matrix(t[0], t[1])
        for i in t[1:]:
            wct = torch.cat((wct, compute_distance_matrix(i, i)))
            for j in t:
                if not torch.equal(i, j): # if j is not i:
                    bct = torch.cat((bct, compute_distance_matrix(i, j)))

        s = torch.reshape(source_features, (int(batch_size / instances), instances, feature_size))
        wcs = compute_distance_matrix(s[0], s[0])
        bcs = compute_distance_matrix(s[0], s[1])
        for i in s[1:]:
            wcs = torch.cat((wcs, compute_distance_matrix(i, i)))
            for j in s:
                if not torch.equal(i, j): # if j is not i:
                    bcs = torch.cat((bcs, compute_distance_matrix(i, j)))

        # We want to modify only target distribution
        bcs = bcs.detach()
        wcs = wcs.detach()

        if self.distance_only:
            return self.mmd_loss(wcs, wct), self.mmd_loss(bcs, bct)

        return self.mmd_loss(wcs, wct), self.mmd_loss(bcs, bct), self.mmd_loss(source_features, target_features)
        #return self.mmd_loss(wcs, wct), self.mmd_loss(bcs, bct), torch.tensor(0)
        #return torch.tensor(0), self.mmd_loss(bcs, bct), torch.tensor(0)
        #return self.mmd_loss(wcs, wct), torch.tensor(0), torch.tensor(0)
