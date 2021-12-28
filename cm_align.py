import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class CMAlign(nn.Module):
    def __init__(self, batch_size=8, num_pos=4, temperature=50, use_gray=False):
        super(CMAlign, self).__init__()
        self.batch_size = batch_size
        self.num_pos = num_pos
        self.criterion = nn.TripletMarginLoss(margin=0.3, p=2.0, reduce=False)
        self.temperature = temperature
        self.use_gray = use_gray

    def _random_pairs(self):
        batch_size = self.batch_size
        num_pos = self.num_pos

        pos = []
        for batch_index in range(batch_size):
            pos_idx = random.sample(list(range(num_pos)), num_pos)
            pos_idx = np.array(pos_idx) + num_pos*batch_index
            pos = np.concatenate((pos, pos_idx))
        pos = pos.astype(int)

        neg = []
        for batch_index in range(batch_size):
            batch_list = list(range(batch_size))
            batch_list.remove(batch_index)

            if batch_size < num_pos :
                batch_idx = np.random.choice(batch_list, num_pos, replace=True)
            else:
                batch_idx = np.random.choice(batch_list, num_pos, replace=False)
            neg_idx = random.sample(list(range(num_pos)), num_pos)

            batch_idx, neg_idx = batch_idx, np.array(neg_idx)
            neg_idx = batch_idx*num_pos + neg_idx
            neg = np.concatenate((neg, neg_idx))
        neg = neg.astype(int)

        return {'pos': pos, 'neg': neg}

    def _dist_pairs(self, feat, labels=None):
        batch_size = self.batch_size
        num_pos = self.num_pos
        n = feat.size(0)
        with torch.no_grad():
            targets = torch.from_numpy(np.tile(np.arange(batch_size).repeat(num_pos), 2))
            if feat.is_cuda:
                target = targets.cuda()


            dist = torch.pow(feat, 2).sum(dim=1, keepdim=True).expand(n, n)
            dist = dist + dist.t()
            dist.addmm_(1, -2, feat, feat.t())
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
            mask = targets.expand(n, n).eq(targets.expand(n, n).t())
            dist_only_pos = dist.clone()
            dist_only_neg = dist.clone()
            pos = []
            neg = []
            for i in range(n):
                dist_only_pos[i][~mask[i]] = -1e10 # set non same distance very low making sure they not selected in finding max
                dist_only_neg[i][mask[i]] = 1e10 # vice versa
                pos.append(dist_only_pos[i].argmax().item())
                neg.append(dist_only_neg[i].argmin().item())

        return {'pos': pos, 'neg': neg}

    def _define_pairs(self, feat):
        #return self._dist_pairs(feat)
        pairs_v = self._random_pairs()
        pos_v, neg_v = pairs_v['pos'], pairs_v['neg']

        pairs_t = self._random_pairs()
        pos_t, neg_t = pairs_t['pos'], pairs_t['neg']

        shift = self.batch_size*self.num_pos # [v;t] shift between first index of v and t in feat

        pos_t += shift
        neg_t += shift

        if self.use_gray :
            pairs_g = self._random_pairs()
            pos_g1, _ = pairs_g['pos'], pairs_g['neg']
            pairs_g = self._random_pairs()
            pos_g2, neg_g = pairs_g['pos'], pairs_g['neg']

            pos_g1 += 2*shift  # g indices comes after t feat
            pos_g2 += 2*shift
            neg_g += 2*shift
            pos_vt = np.random.choice(np.concatenate((pos_v, pos_t)), shift, replace=False)  # for positive of g, it selects random choice of v and t
            pos = np.concatenate((pos_g1, pos_g2, pos_vt))  # makes positive indices be selected from cross modalities
            neg = np.concatenate((neg_v, neg_t, neg_g))
        else:
            pos = np.concatenate((pos_t, pos_v))  # makes positive indices be selected from cross modalities
            neg = np.concatenate((neg_v, neg_t))

        return {'pos': pos, 'neg': neg}

    def feature_similarity(self, feat_q, feat_k):
        batch_size, fdim, h, w = feat_q.shape
        feat_q = feat_q.view(batch_size, fdim, -1)
        feat_k = feat_k.view(batch_size, fdim, -1)

        feature_sim = torch.bmm(F.normalize(feat_q, dim=1).permute(0,2,1), F.normalize(feat_k, dim=1))
        return feature_sim

    def matching_probability(self, feature_sim):
        M, _ = feature_sim.max(dim=-1, keepdim=True)
        feature_sim = feature_sim - M # for numerical stability
        exp = torch.exp(self.temperature*feature_sim)
        exp_sum = exp.sum(dim=-1, keepdim=True)
        return exp / exp_sum

    def soft_warping(self, matching_pr, feat_k):
        batch_size, fdim, h, w = feat_k.shape
        feat_k = feat_k.view(batch_size, fdim, -1)
        feat_warp = torch.bmm(matching_pr, feat_k.permute(0,2,1))
        feat_warp = feat_warp.permute(0,2,1).view(batch_size, fdim, h, w)

        return feat_warp

    def reconstruct(self, mask, feat_warp, feat_q):
        return mask*feat_warp + (1.0-mask)*feat_q

    def compute_mask(self, feat):
        batch_size, fdim, h, w = feat.shape
        norms = torch.norm(feat, p=2, dim=1).view(batch_size, h*w)

        norms -= norms.min(dim=-1, keepdim=True)[0]
        norms /= norms.max(dim=-1, keepdim=True)[0] + 1e-12
        mask = norms.view(batch_size, 1, h, w)

        return mask.detach()

    def compute_comask(self, matching_pr, mask_q, mask_k):
        batch_size, mdim, h, w = mask_q.shape
        mask_q = mask_q.view(batch_size, -1, 1)
        mask_k = mask_k.view(batch_size, -1, 1)
        comask = mask_q * torch.bmm(matching_pr, mask_k)

        comask = comask.view(batch_size, -1)
        comask -= comask.min(dim=-1, keepdim=True)[0]
        comask /= comask.max(dim=-1, keepdim=True)[0] + 1e-12
        comask = comask.view(batch_size, mdim, h, w)

        return comask.detach()

    def forward(self, feat_v, feat_t, out_feat=None, feat_g=None):
        feat = torch.cat([feat_v, feat_t], dim=0)
        if self.use_gray and feat_g is not None:
            feat = torch.cat([feat, feat_g], dim=0)
        mask = self.compute_mask(feat)
        batch_size, fdim, h, w = feat.shape

        pairs = self._define_pairs(out_feat)
        pos_idx, neg_idx = pairs['pos'], pairs['neg']

        # positive
        feat_target_pos = feat[pos_idx]
        feature_sim = self.feature_similarity(feat, feat_target_pos)
        matching_pr = self.matching_probability(feature_sim)

        comask_pos = self.compute_comask(matching_pr, mask, mask[pos_idx])
        feat_warp_pos = self.soft_warping(matching_pr, feat_target_pos)
        feat_recon_pos = self.reconstruct(mask, feat_warp_pos, feat)

        # negative
        feat_target_neg = feat[neg_idx]
        feature_sim = self.feature_similarity(feat, feat_target_neg)
        matching_pr = self.matching_probability(feature_sim)

        feat_warp = self.soft_warping(matching_pr, feat_target_neg)
        feat_recon_neg = self.reconstruct(mask, feat_warp, feat)

        loss = torch.mean(comask_pos * self.criterion(feat, feat_recon_pos, feat_recon_neg))

        return {'feat': feat_recon_pos, 'loss': loss}

