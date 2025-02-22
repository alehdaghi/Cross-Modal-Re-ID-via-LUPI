from __future__ import print_function
import argparse
import time
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from loss import pdist_np
from model import embed_net
from modelPart import embed_net as embed_net_part
from utils import *
import pdb
import random
from datetime import datetime
random.seed(datetime.now())

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', '-d', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', '-a', default='resnet50', type=str,
                    help='network baseline: resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=28, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='base', type=str,
                    metavar='m', help='method type: base or awg')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet losses margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--uni', default=0, type=int,
                    help='0: two modality, 1: Only Vis 2: Only Ir 3: Only Gray used in training')
parser.add_argument('--mode', default='all', type=str, help='all or indoor for sysu')
parser.add_argument('--tvsearch', action='store_true', help='whether thermal to visible search on RegDB')
parser.add_argument('--multi', dest='multi_shot', help='multi shot for testing (10 images for each id in gallery instead of 1) ', action='store_true')
parser.add_argument('--cont', dest='cont_loss', help='use Contrastive Loss', action='store_true')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

dataset = args.dataset
if dataset == 'sysu':
    data_path = '../Datasets/SYSU-MM01/'
    n_class = 395
    if args.uni == 0:
        test_mode = [1, 2]  # thermal to visible
    else:
        test_mode = [3, 3]  # thermal to visible

elif dataset =='regdb':
    data_path = '../Datasets/RegDB/'
    n_class = 395
    test_mode = [2, 1]
 
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0 

print('==> Building model..')
if args.method =='base':
    net = embed_net_part(n_class, no_local= 'off', gm_pool =  'off', arch=args.arch)
    # print(net.count_params())
else:
    net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch, use_contrast=args.cont_loss)
pool_dim = net.getPoolDim()
if args.arch == 'resnet18':
    pool_dim = 512
print(net.thermal_module.count_params())
net.to(device)    
cudnn.benchmark = True

checkpoint_path = args.model_path

if args.method =='id':
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomCrop((args.img_h,args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h,args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()



def extract_gall_feat(gall_loader):
    net.eval()
    print ('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat_pool = np.zeros((ngall, pool_dim))
    gall_feat_fc = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label, cam) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input, x3=input, modal=test_mode[0])
            gall_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
            gall_feat_fc[ptr:ptr+batch_num,: ]   = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return gall_feat_pool, gall_feat_fc
    
def extract_query_feat(query_loader):
    net.eval()
    print ('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat_pool = np.zeros((nquery, pool_dim))
    query_feat_fc = np.zeros((nquery, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label, cam ) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat_pool, feat_fc = net(input, input, x3=input, modal=test_mode[1])
            query_feat_pool[ptr:ptr+batch_num,: ] = feat_pool.detach().cpu().numpy()
            query_feat_fc[ptr:ptr+batch_num,: ]   = feat_fc.detach().cpu().numpy()
            ptr = ptr + batch_num         
    print('Extracting Time:\t {:.3f}'.format(time.time()-start))
    return query_feat_pool, query_feat_fc

N = 10
print('==> Resuming from checkpoint..')
if len(args.resume) > 0:
    model_path = args.resume
    # model_path = checkpoint_path + 'sysu_awg_p4_n8_lr_0.1_seed_0_best.t'
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'

              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

all_cmc, all_mAP, all_mINP = [], [], []
all_cmc_pool, all_mAP_pool, all_mINP_pool = [], [], []


if dataset == 'sysu':

    # testing set
    if args.uni == 1:
        args.mode = 'Vis'
    elif args.uni == 2:
        args.mode = 'Ir'
    elif args.uni == 3:
        args.mode = 'Gray'
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode,
                                                          trial=0,  single_shot=(not args.multi_shot))

    nquery = len(query_label)
    ngall = len(gall_label)
    print("Dataset statistics:")
    print("  ------------------------------")
    print("  subset   | # ids | # images")
    print("  ------------------------------")
    print("  query    | {:5d} | {:8d}".format(len(np.unique(query_label)), nquery))
    print("  gallery  | {:5d} | {:8d}".format(len(np.unique(gall_label)), ngall))
    print("  ------------------------------")

    queryset = TestData(query_img, query_label, query_cam, transform=transform_test, img_size=(args.img_w, args.img_h), colorToGray=args.uni == 3)
    query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
    print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

    query_feat_pool, query_feat_fc = extract_query_feat(query_loader)

    for trial in range(N):

        trial_gallset = TestData(gall_img, gall_label, gall_cam, transform=transform_test, img_size=(args.img_w, args.img_h), colorToGray=args.uni == 3)
        trial_gall_loader = data.DataLoader(trial_gallset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        ngall = len(trial_gallset)
        gall_feat_pool, gall_feat_fc = extract_gall_feat(trial_gall_loader)

        # gall_feat_pool, query_feat_pool, gall_feat_fc, query_feat_fc \
        #     = query_feat_pool, gall_feat_pool, query_feat_fc, gall_feat_fc
        # query_label, query_cam, gall_label, gall_cam = \
        #     gall_label, gall_cam, query_label, query_cam

        # pool5 feature
        start = time.time()
        distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
        #distmat_pool = pdist_np(query_feat_pool, gall_feat_pool)
        gall_label = trial_gallset.test_label
        gall_cam = trial_gallset.test_cam
        cmc_pool, mAP_pool, mINP_pool = eval_sysu(-distmat_pool, query_label, gall_label, query_cam, gall_cam)

        # fc feature
        distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)

        all_cmc.append(cmc)
        all_mAP.append(mAP)
        all_mINP.append(mINP)
        all_cmc_pool.append(cmc_pool)
        all_mAP_pool.append(mAP_pool)
        all_mINP_pool.append(mINP_pool)

        print('Evaluating Time:\t {:.3f}'.format(time.time() - start))
        print('Test Trial: {}'.format(trial))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'POOL: Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))
        #if args.multi_shot:
        #    N = 1
        #    break
        gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=trial+1, single_shot=(not args.multi_shot))


elif dataset == 'regdb':

    for trial in range(N):
        test_trial = trial +1
        # #model_path = checkpoint_path +  args.resume
        # model_path = checkpoint_path + 'regdb_awg_p4_n8_lr_0.1_seed_0_trial_{}_best.t'.format(test_trial)
        # if os.path.isfile(model_path):
        #     print('==> loading checkpoint {}'.format(args.resume))
        #     checkpoint = torch.load(model_path)
        #     net.load_state_dict(checkpoint['net'])

        # training set
        trainset = RegDBData(data_path, test_trial, transform=transform_train)
        # generate the idx of each person identity
        color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_ir_label)

        # testing set
        query_img, query_label, query_cam = process_test_regdb(data_path, trial=test_trial, modal='visible')
        gall_img, gall_label, gall_cam = process_test_regdb(data_path, trial=test_trial, modal='thermal')

        gallset = TestData(gall_img, gall_label, gall_cam, transform=transform_test, img_size=(args.img_w, args.img_h))
        gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

        nquery = len(query_label)
        ngall = len(gall_label)

        queryset = TestData(query_img, query_label, query_cam, transform=transform_test, img_size=(args.img_w, args.img_h))
        query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=4)
        print('Data Loading Time:\t {:.3f}'.format(time.time() - end))


        query_feat_pool, query_feat_fc = extract_query_feat(query_loader)
        gall_feat_pool,  gall_feat_fc = extract_gall_feat(gall_loader)

        if args.tvsearch:
            # pool5 feature
            distmat_pool = np.matmul(gall_feat_pool, np.transpose(query_feat_pool))
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, gall_label, query_label)

            # fc feature
            distmat = np.matmul(gall_feat_fc , np.transpose(query_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat,gall_label,  query_label )
        else:
            # pool5 feature
            distmat_pool = np.matmul(query_feat_pool, np.transpose(gall_feat_pool))
            cmc_pool, mAP_pool, mINP_pool = eval_regdb(-distmat_pool, query_label, gall_label)

            # fc feature
            distmat = np.matmul(query_feat_fc, np.transpose(gall_feat_fc))
            cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)


        if trial == 0:
            all_cmc = cmc
            all_mAP = mAP
            all_mINP = mINP
            all_cmc_pool = cmc_pool
            all_mAP_pool = mAP_pool
            all_mINP_pool = mINP_pool
        else:
            all_cmc = all_cmc + cmc
            all_mAP = all_mAP + mAP
            all_mINP = all_mINP + mINP
            all_cmc_pool = all_cmc_pool + cmc_pool
            all_mAP_pool = all_mAP_pool + mAP_pool
            all_mINP_pool = all_mINP_pool + mINP_pool

        print('Test Trial: {}'.format(trial))
        print(
            'FC:     Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_pool[0], cmc_pool[4], cmc_pool[9], cmc_pool[19], mAP_pool, mINP_pool))


cmc = [np.asarray(all_cmc).mean(axis=0), np.asarray(all_cmc).var(axis=0)]
mAP = [np.asarray(all_mAP).mean(axis=0), np.asarray(all_mAP).var(axis=0)]
mINP = [np.asarray(all_mINP).mean(axis=0), np.asarray(all_mINP).var(axis=0)]

cmc_pool = [np.asarray(all_cmc_pool).mean(axis=0), np.asarray(all_cmc_pool).var(axis=0)]
mAP_pool = [np.asarray(all_mAP_pool).mean(axis=0), np.asarray(all_mAP_pool).var(axis=0)]
mINP_pool = [np.asarray(all_mINP_pool).mean(axis=0), np.asarray(all_mINP_pool).var(axis=0)]

print('All Average:')
print('FC:     Rank-1: {:.2%}±{:.2%} | Rank-5: {:.2%}±{:.2%} | Rank-10: {:.2%}±{:.2%}| Rank-20: {:.2%}± {:.2%}| mAP: {:.2%}±{:.2%}| mINP: {:.2%}±{:.2%}'.format(
        cmc[0][0], cmc[1][0], cmc[0][4],cmc[1][4], cmc[0][9],cmc[1][9], cmc[0][19],cmc[1][9], mAP[0],mAP[1], mINP[0], mINP[1]))
print('POOL:   Rank-1: {:.2%}±{:.2%} | Rank-5: {:.2%}±{:.2%} | Rank-10: {:.2%}±{:.2%}| Rank-20: {:.2%}± {:.2%}| mAP: {:.2%}±{:.2%}| mINP: {:.2%}±{:.2%}'.format(
        cmc_pool[0][0], cmc_pool[1][0], cmc_pool[0][4],cmc_pool[1][4], cmc_pool[0][9],cmc_pool[1][9], cmc_pool[0][19],cmc_pool[1][9], mAP_pool[0],mAP_pool[1], mINP_pool[0], mINP_pool[1]))
