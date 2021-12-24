import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd.function import Function
from torch.autograd import Variable


class OriTripletLoss(nn.Module):
    """Triplet losses with hard positive/negative mining.
    
    Reference:
    Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/loss/triplet.py.
    
    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, batch_size, margin=0.3):
        super(OriTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge losses
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


# Adaptive weights
def softmax_weights(dist, mask):
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6  # avoid division by zero
    W = torch.exp(diff) * mask / Z
    return W


def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet'."""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=False):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        # shape [N, N]
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        # `dist_ap` means distance(anchor, positive)
        # both `dist_ap` and `relative_p_inds` with shape [N, 1]
        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        # compute accuracy
        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm_(1, -2, emb1, emb2.t())
    # dist_mtx = dist_mtx.clamp(min = 1e-12)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx


def pdist_np(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using cpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = np.square(emb1).sum(axis=1)[..., np.newaxis]
    emb2_pow = np.square(emb2).sum(axis=1)[np.newaxis, ...]
    dist_mtx = -2 * np.matmul(emb1, emb2.T) + emb1_pow + emb2_pow
    # dist_mtx = np.sqrt(dist_mtx.clip(min = 1e-12))
    return dist_mtx


class RankingLoss:

    def __init__(self):
        pass

    def _label2similarity(sekf, label1, label2):
        '''
        compute similarity matrix of label1 and label2
        :param label1: torch.Tensor, [m]
        :param label2: torch.Tensor, [n]
        :return: torch.Tensor, [m, n], {0, 1}
        '''
        m, n = len(label1), len(label2)
        l1 = label1.view(m, 1).expand([m, n])
        l2 = label2.view(n, 1).expand([n, m]).t()
        similarity = l1 == l2
        return similarity

    def _batch_hard(self, mat_distance, mat_similarity, more_similar='smaller'):

        if more_similar is 'smaller':
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (1 - mat_similarity), dim=1,
                                                descending=True)
            hard_p = sorted_mat_distance[:, 0]
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (mat_similarity), dim=1, descending=False)
            hard_n = sorted_mat_distance[:, 0]
            return hard_p, hard_n

        elif more_similar is 'larger':
            sorted_mat_distance, _ = torch.sort(mat_distance + (9999999.) * (1 - mat_similarity), dim=1,
                                                descending=False)
            hard_p = sorted_mat_distance[:, 0]
            sorted_mat_distance, _ = torch.sort(mat_distance + (-9999999.) * (mat_similarity), dim=1, descending=True)
            hard_n = sorted_mat_distance[:, 0]
            return hard_p, hard_n


class TripletLoss(RankingLoss):
    '''
    Compute Triplet losses augmented with Batch Hard
    Details can be seen in 'In defense of the Triplet Loss for Person Re-Identification'
    '''

    def __init__(self, margin, metric):
        '''
        :param margin: float or 'soft', for MarginRankingLoss with margin and soft margin
        :param bh: batch hard
        :param metric: l2 distance or cosine distance
        '''
        self.margin = margin
        self.margin_loss = nn.MarginRankingLoss(margin=margin)
        self.metric = metric

    def __call__(self, emb1, emb2, emb3, label1, label2, label3):
        '''

		:param emb1: torch.Tensor, [m, dim]
		:param emb2: torch.Tensor, [n, dim]
		:param label1: torch.Tensor, [m]
		:param label2: torch.Tensor, [b]
		:return:
		'''

        if self.metric == 'cosine':
            mat_dist = cosine_dist(emb1, emb2)
            mat_sim = self._label2similarity(label1, label2)
            hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

            mat_dist = cosine_dist(emb1, emb3)
            mat_sim = self._label2similarity(label1, label3)
            _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='larger')

            margin_label = -torch.ones_like(hard_p)

        elif self.metric == 'euclidean':
            mat_dist = euclidean_dist(emb1, emb2)
            mat_sim = self._label2similarity(label1, label2)
            hard_p, _ = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

            mat_dist = euclidean_dist(emb1, emb3)
            mat_sim = self._label2similarity(label1, label3)
            _, hard_n = self._batch_hard(mat_dist, mat_sim.float(), more_similar='smaller')

            margin_label = torch.ones_like(hard_p)

        return self.margin_loss(hard_n, hard_p, margin_label)


class HetroCenterLoss(nn.Module):
    """Triplet losses with hard positive/negative mining.

    Reference:
    Zhu et al. Hetero-Center Loss for Cross-Modality Person Re-Identification. arXiv:1910.09830.
    Code imported from https://github.com/98zyx/Hetero-center-loss-for-cross-modality-person-re-id/blob/master/heterogeneity_loss.py.

    Args:
    - margin (float): margin for triplet.
    - dist_type (string): distance metric type (l2, l1 or cosine) .
    """

    def __init__(self, margin=0.01, dist_type='l2'):
        super(HetroCenterLoss, self).__init__()
        self.margin = margin
        self.dist_type = dist_type
        if dist_type == 'l2':
            self.dist = nn.MSELoss(reduction='sum')
        if dist_type == 'cos':
            self.dist = nn.CosineSimilarity(dim=0)
        if dist_type == 'l1':
            self.dist = nn.L1Loss()

    def forward(self, feat1, feat2, label1, label2):
        feat_size = feat1.size()[1]
        feat_num = feat1.size()[0]
        label_num = len(label1.unique())
        # min1 = feat1.min(0, keepdim=True)[0]
        # max1 = feat1.max(0, keepdim=True)[0]
        # min2 = feat2.min(0, keepdim=True)[0]
        # max2 = feat2.max(0, keepdim=True)[0]


        center1 = torch.mean(feat1.view(label_num, -1, feat_size), dim=1)
        center2 = torch.mean(feat2.view(label_num, -1, feat_size), dim=1)
        # losses = Variable(.cuda())
        #dist = torch.tensor(0.0, requires_grad=True, device=label1.device)
        # for i in range(label_num):
        #     #center1 = 2*(torch.mean(feat1[i], dim=0)-min1)/max1 - 1
        #     #center2 = 2*(torch.mean(feat2[i], dim=0)-min2)/max2 - 1
        #     center1 = torch.mean(feat1_c[i], dim=0)
        #     center2 = torch.mean(feat2_c[i], dim=0)
        #     if self.dist_type == 'l2' or self.dist_type == 'l1':
        #         if i == 0:
        #             dist = max(0, self.dist(center1, center2)/feat_size - self.margin)
        #         else:
        #             dist += max(0, self.dist(center1, center2)/feat_size - self.margin)
        #     elif self.dist_type == 'cos':
        #         if i == 0:
        #             dist = max(0, 1 - self.dist(center1, center2) - self.margin)
        #         else:
        #             dist += max(0, 1 - self.dist(center1, center2) - self.margin)
        #dist /= label_num
        return self.dist(center1, center2)/feat_size


class HcTripletLoss(nn.Module):
    """Hetero-center-triplet-losses-for-VT-Re-ID.

    Reference:
    Parameter Sharing Exploration and Hetero center triplet losses for VT Re-ID,TMM.
    Code imported from https://github.com/hijune6/Hetero-center-triplet-loss-for-VT-Re-ID/blob/main/loss.py.

    Args:
    - margin (float): margin for triplet.
    """

    def __init__(self, margin=0.3):
        super(HcTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, feats, labels):
        """
        Args:
        - inputs: feature matrix with shape (batch_size, feat_dim)
        - targets: ground truth labels with shape (num_classes)
        """
        label_uni = labels.unique()
        targets = torch.cat([label_uni, label_uni])
        label_num = len(label_uni)
        feat = feats.chunk(label_num * 2, 0)
        center = []
        for i in range(label_num * 2):
            center.append(torch.mean(feat[i], dim=0, keepdim=True))
        inputs = torch.cat(center)

        n = inputs.size(0)

        # Compute pairwise distance, replace by the official when merged
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(1, -2, inputs, inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # For each anchor, find the hardest positive and negative
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        # Compute ranking hinge losses
        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)

        # compute accuracy
        correct = torch.ge(dist_an, dist_ap).sum().item()
        return loss, correct


class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy losses with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, num_classes, epsilon=0.1, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss



class CenterLoss(nn.Module):
    """Center losses.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=751, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), "features.size(0) is not equal to labels.size(0)"

        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = []
        for i in range(batch_size):
            value = distmat[i][mask[i]]
            value = value.clamp(min=1e-12, max=1e+12)  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss


def cosine_dist(x, y):
    '''
	:param x: torch.tensor, 2d
	:param y: torch.tensor, 2d
	:return:
	'''

    bs1 = x.size()[0]
    bs2 = y.size()[0]

    frac_up = torch.matmul(x, y.transpose(0, 1))
    frac_down = (torch.sqrt(torch.sum(torch.pow(x, 2), 1))).view(bs1, 1).repeat(1, bs2) * \
                (torch.sqrt(torch.sum(torch.pow(y, 2), 1))).view(1, bs2).repeat(bs1, 1)
    cosine = frac_up / frac_down

    return cosine


def euclidean_dist(x, y):
    """
	Args:
	  x: pytorch Variable, with shape [m, d]
	  y: pytorch Variable, with shape [n, d]
	Returns:
	  dist: pytorch Variable, with shape [m, n]
	"""
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def categorical_cross_entropy(y_pred, y_true):
    y_pred = torch.clamp(y_pred, 1e-9, 1 - 1e-9)
    return -(y_true * torch.log(y_pred)).sum(dim=1).mean()
