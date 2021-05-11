import time

import torch
import torch.nn as nn
#from testRegDB import RegDBData
from data_loader import SYSUData
from model import *
import torchvision.transforms as transforms
import numpy as np
from sklearn.datasets import load_digits
from scipy.spatial.distance import pdist
from sklearn.manifold._t_sne import _joint_probabilities
from scipy import linalg
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import squareform
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
from torch.autograd import Variable
import pandas as pd
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from PIL import Image


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((288, 144)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((288, 144)),
    transforms.ToTensor(),
    normalize,
])

sns.set(rc={'figure.figsize':(11.7,8.27)})

MACHINE_EPSILON = np.finfo(np.double).eps
n_components = 2
perplexity = 30
pool_dim = 2048

def fit(X):
    n_samples = X.shape[0]

    # Compute euclidean distance
    distances = pairwise_distances(X, metric='euclidean', squared=True)

    # Compute joint probabilities p_ij from distances.
    P = _joint_probabilities(distances=distances, desired_perplexity=perplexity, verbose=False)

    # The embedding is initialized with iid samples from Gaussians with standard deviation 1e-4.
    X_embedded = 1e-4 * np.random.mtrand._rand.randn(n_samples, n_components).astype(np.float32)

    # degrees_of_freedom = n_components - 1 comes from
    # "Learning a Parametric Embedding by Preserving Local Structure"
    # Laurens van der Maaten, 2009.
    degrees_of_freedom = max(n_components - 1, 1)

    return _tsne(P, degrees_of_freedom, n_samples, X_embedded=X_embedded)

def _tsne(P, degrees_of_freedom, n_samples, X_embedded):
    params = X_embedded.ravel()
    obj_func = _kl_divergence
    params = _gradient_descent(obj_func, params, [P, degrees_of_freedom, n_samples, n_components])
    X_embedded = params.reshape(n_samples, n_components)
    return X_embedded

def _kl_divergence(params, P, degrees_of_freedom, n_samples, n_components):
    X_embedded = params.reshape(n_samples, n_components)

    dist = pdist(X_embedded, "sqeuclidean")
    dist /= degrees_of_freedom
    dist += 1.
    dist **= (degrees_of_freedom + 1.0) / -2.0
    Q = np.maximum(dist / (2.0 * np.sum(dist)), MACHINE_EPSILON)

    # Kullback-Leibler divergence of P and Q
    kl_divergence = 2.0 * np.dot(P, np.log(np.maximum(P, MACHINE_EPSILON) / Q))

    # Gradient: dC/dY
    grad = np.ndarray((n_samples, n_components), dtype=params.dtype)
    PQd = squareform((P - Q) * dist)
    for i in range(n_samples):
        grad[i] = np.dot(np.ravel(PQd[i], order='K'),
                         X_embedded[i] - X_embedded)
    grad = grad.ravel()
    c = 2.0 * (degrees_of_freedom + 1.0) / degrees_of_freedom
    grad *= c


    return kl_divergence, grad

def _gradient_descent(obj_func, p0, args, it=0, n_iter=1000,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7):
    p = p0.copy().ravel()
    update = np.zeros_like(p)
    gains = np.ones_like(p)
    error = np.finfo(np.float).max
    best_error = np.finfo(np.float).max
    best_iter = i = it

    for i in range(it, n_iter):
        error, grad = obj_func(p, *args)
        grad_norm = linalg.norm(grad)
        inc = update * grad < 0.0
        dec = np.invert(inc)
        gains[inc] += 0.2
        gains[dec] *= 0.8
        np.clip(gains, min_gain, np.inf, out=gains)
        grad *= gains
        update = momentum * update - learning_rate * grad
        p += update
        print("[t-SNE] Iteration %d: error = %.7f,"
              " gradient norm = %.7f"
              % (i + 1, error, grad_norm))

        if error < best_error:
            best_error = error
            best_iter = i
        elif i - best_iter > n_iter_without_progress:
            break

        if grad_norm <= min_grad_norm:
            break
    return p



L = 10
data_path = '../Datasets/SYSU-MM01/'
n_class = 395
query_img, query_label, query_cam = process_query_sysu(data_path,
                                                      file_path='exp/train_id.txt')
gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, trial=0, single_shot=False,
                                                      file_path='exp/train_id.txt')
nquery = len(query_label)
ngall = len(gall_label)
img_size = (144,288)

def extractFeat(imgs, labels, cams, TEST_TYPE ):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_class = 296
    net = embed_net(n_class, no_local='off', gm_pool='off', arch='resnet50')
    resume = './save_model/sysu_base_p4_n10_lr_0.1_seed_0_gray_epoch_80.t'
    #resume = 'save_model/sysu_base_p6_n8_lr_0.1_seed_2_epoch_80.t'
    pool_dim = net.getPoolDim()
    #resume = './save_model/sysu_base_p4_n8_lr_0.1_seed_0_first.t'
    checkpoint = torch.load(resume)
    net.to(device)
    net.load_state_dict(checkpoint['net'])
    net.eval()

    with torch.no_grad():

        XX = np.empty((0, pool_dim))
        YY = np.empty(0, np.int)
        CC = np.empty(0, np.int)

        m = {i: [] for i in np.unique(labels)}
        for i, y in enumerate(labels):
            m[y] = np.append(m[y], int(i))
        #Ls = np.random.choice(data.getLabels(), L)
        Ls = np.unique(labels)[0:L]
        for i in Ls:
            X = torch.Tensor()
            for j in m[i]:
                j = int(j)
                img = Image.open(imgs[j])
                img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
                img = np.array(img)
                if TEST_TYPE == 3:
                    img = np.dot(img[..., :3], [0.299, 0.587, 0.144]).astype(img.dtype)
                    img = np.stack((img,) * 3, axis=-1)

                x = transform_test(img)
                y1 = labels[j]
                c1 = cams[j]
                X = torch.cat((X, x.unsqueeze(0)), dim=0)
                YY = np.append(YY, [int(y1)], axis=0)
                CC = np.append(CC, [int(c1)], axis=0)


            X = Variable(X.cuda())

            #x1 = x1.unsqueeze(0)
            #x2 = x2.unsqueeze(0)
            i = 0
            while i < len(X):
                j = min(i+20, len(X))
                res = net(X[i:j], X[i:j], x3 = X[i:j], modal=TEST_TYPE)
                XX = np.append(XX, res[0].cpu().numpy() , axis=0)
                i = j

    return XX, YY, CC




USE_NET = True
USE_OTHER = True
if USE_NET:
    XQ, yQ, cQ = extractFeat(query_img, query_label, query_cam, 2)
    #XQ, yQ, cQ = extractFeat(gall_img, gall_label, gall_cam, 1)
    XG, yG, cG = extractFeat(gall_img, gall_label, gall_cam, 1)
    X = np.append(XQ, XG, axis=0)
    y = np.append(yQ, yG, axis=0)
    c = np.append(cQ, cG, axis=0)
    X_embedded = fit(X)
    np.save('X.npy', X_embedded)
    np.save('y.npy', y)
    np.save('c.npy', c)
    i1 = np.unique(yQ, return_counts=True)[1]
    i2 = np.unique(yG, return_counts=True)[1]

    f1 = np.split(XQ, np.cumsum(i1[:-1]))
    f2 = np.split(XG, np.cumsum(i2[:-1]))
    c1 = np.asarray([np.mean(f, axis=0) for f in f1])
    c2 = np.asarray([np.mean(f, axis=0) for f in f2])


else:
    if USE_OTHER:
        X = np.load('/home/mahdi/PycharmProjects/DGTL-for-VT-ReID/f.npy')
        y = np.load('/home/mahdi/PycharmProjects/DGTL-for-VT-ReID/p.npy')
        c = np.load('/home/mahdi/PycharmProjects/DGTL-for-VT-ReID/c.npy')
        Ls = np.unique(y)[0:L]
        X = X[np.isin(y, Ls)]
        c = c[np.isin(y, Ls)]
        y = y[np.isin(y, Ls)]
        X_embedded = fit(X)
    else:
        X_embedded = np.load('X.npy')
        y = np.load('y.npy')
        c = np.load('c.npy')


palette = sns.color_palette("muted", L)
df = pd.DataFrame(columns=['x','y','id','cR', 'cI'])
df['x'] = X_embedded[:,0]
df['y'] = X_embedded[:,1]
df['id'] = y
df['c'] = c

sns.set_theme()
p = sns.hls_palette(L, h=.5)
plt.title("Visualize features of "+str(L) +" identity (Resnet50, Part-Att)" )
palette = ['green', 'orange', 'gray', 'brown', 'dodgerblue', 'black']
g=sns.scatterplot(data=df, x='x', y='y', style= 'id', hue='c', palette=palette)
plt.legend(bbox_to_anchor=(1.01, 1),borderaxespad=0)
plt.show()
