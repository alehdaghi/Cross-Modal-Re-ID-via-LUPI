from __future__ import print_function
import argparse
import sys
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
#from transformer.make_model import *
from transformer.vit_for_small_dataset import ViT
from torch.optim.lr_scheduler import StepLR

from utils import *
from loss import *
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1 , type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
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
parser.add_argument('--workers', default=0, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=4, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=48, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str,
                    metavar='m', help='method type: base or agw')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet losses margin')
parser.add_argument('--num_pos', default=8, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--depth', default=6, type=int,
                    help='Depth of transformer (layers)')

parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='0', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--use_gray', dest='use_gray', help='use gray as 3rd modality', action='store_true')
parser.add_argument('--separate_batch_norm', dest='separate_batch_norm', help='separate batch norm layers only in first layers',
                    action='store_true')

seed = 23
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True

parser.set_defaults(use_gray=False)
parser.set_defaults(separate_batch_norm=False)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '../Datasets/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = '../Datasets/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
suffix = suffix + '_sViT_p{}_n{}_lr_{}_depth_{}'.format(args.num_pos, args.batch_size, args.lr, args.depth)


if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if args.use_gray:
    suffix = suffix + '_gray'

if args.separate_batch_norm:
    suffix = suffix + '_sepBatch'


suffix = suffix + '_trans'



if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_ir_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0, single_shot=False)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_ir_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, gall_cam, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, query_cam, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_ir_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')

# patch_net = embed_net(n_class, no_local= 'off', gm_pool =  'off', arch='resnet18')
# checkpoint = torch.load("save_model/sysu_base_p8_n16_lr_0.1_seed_0_gray_arch18_best.t")
# patch_net.load_state_dict(checkpoint['net'])

net = ViT(
    image_size = (args.img_w, args.img_h),
    patch_size = 12,
    num_classes = n_class,
    dim = 1024,
    depth = args.depth,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

net.to(device)
cudnn.benchmark = True
print(net.count_params())
pool_dim = net.feat_dim
if args.arch == 'resnet18':
    pool_dim = 512
if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {} Rank-1: {:.2%} mAP: {:.2%})'
              .format(args.resume, checkpoint['epoch'],checkpoint['cmp'], checkpoint['mAP']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define losses function
criterion_id = nn.CrossEntropyLoss()
if args.method == 'agw':
    criterion_tri = TripletLoss_WRT()
else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri= OriTripletLoss(batch_size=loader_batch, margin=args.margin)

cross_triplet_creiteron = TripletLoss(0.3, 'euclidean')
reconst_loss = nn.MSELoss()
hetro_loss = HetroCenterLoss()
hctriplet = HcTripletLoss(margin=0.8)
center_criterion = CenterLoss(num_classes=n_class, feat_dim=pool_dim, use_gpu=device == 'cuda')

criterion_id.to(device)
criterion_tri.to(device)
cross_triplet_creiteron.margin_loss.to(device)
reconst_loss.to(device)

lr = 3e-5
gamma = 0.7
seed = 42
ignored_params = list(map(id, net.normModule.parameters())) \
                 + list(map(id, net.classifier.parameters()))
ids = set(map(id, net.parameters()))
params = filter(lambda p: id(p) in ids, net.parameters())
base_params = filter(lambda p: id(p) not in ignored_params, params)

optimizer = optim.SGD([
    {'params': base_params, 'lr': 0.1 * args.lr},
    {'params': net.normModule.parameters(), 'lr': args.lr},
    {'params': net.classifier.parameters(), 'lr': args.lr}],
    weight_decay=5e-4, momentum=0.9, nesterov=True)

# scheduler
#scheduler = StepLR(optimizer, step_size=1, gamma=gamma)


# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def train(epoch):

    current_lr = adjust_learning_rate(optimizer, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    tri_loss = AverageMeter()
    gray_loss = AverageMeter()
    center_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, input3, label1, label2, _, cam1, cam2) in enumerate(trainloader):

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())
        if args.use_gray:
            labels = torch.cat((label1, label2, label1), 0)
            input3 = Variable(input3.cuda())
        else:
            labels = torch.cat((label1, label2), 0)
            input3 = None

        labels = Variable(labels.cuda())
        cams = Variable((torch.cat((cam1, cam2), 0)).cuda())
        data_time.update(time.time() - end)


        feat, out0 = net(input1, input2, cam_label=cams)
        loss_color2gray = torch.tensor(0.0, requires_grad=True, device=device)

        color_label, thermal_label = torch.split(labels, label1.shape[0])
        loss_tri = []
        loss_id = []

        color_feat, thermal_feat = torch.split(feat, label1.shape[0])
        loss_tri_color = cross_triplet_creiteron(color_feat, thermal_feat, thermal_feat,
                                             color_label, thermal_label, thermal_label)
        loss_tri_thermal = cross_triplet_creiteron(thermal_feat, color_feat, color_feat,
                                               thermal_label, color_label, color_label)
        loss_tri = (loss_tri_color + loss_tri_thermal) / 2
        loss_id = criterion_id(out0, labels)



        #loss_tri, batch_acc = criterion_tri(feat, labels)
        #loss_center = hetro_loss(color_feat, thermal_feat, color_label, thermal_label)
        #l1, _ = hctriplet(feat, labels)


        #correct += (batch_acc / 2)
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item() )

        loss = loss_id + loss_tri

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        gray_loss.update(loss_color2gray.item(), 2 * input1.size(0))
        #center_loss.update(loss_center.item(), 2 * input1.size(0))
        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {now} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'GLoss: {gray_loss.val:.4f} ({gray_loss.avg:.4f}) '
                  'CLoss: {center_loss.val:.4f} ({center_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, now=time_now(), batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss, gray_loss=gray_loss, center_loss=center_loss))
            sys.stdout.flush()

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('gray_loss', gray_loss.avg, epoch)
    writer.add_scalar('center_loss', center_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)



def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, pool_dim))
    gall_feat_att = np.zeros((ngall, pool_dim))
    with torch.no_grad():
        for batch_idx, (input, label, cam) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, x3=input, modal=test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, pool_dim))
    query_feat_att = np.zeros((nquery, pool_dim))
    time_inference = 0
    with torch.no_grad():
        for batch_idx, (input, label, cam) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            start1 = time.time()
            feat, feat_att = net(input, input, x3=input, modal=test_mode[1])
            time_inference += (time.time() - start1)
            #print('Extracting Time:\t {:.3f} len={:d}'.format(time.time() - start1, len(input)))

            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time_inference))
    #exit(0)
    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP      = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att  = eval_regdb(-distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    writer.add_scalar('rank1_att', cmc_att[0], epoch)
    writer.add_scalar('mAP_att', mAP_att, epoch)
    writer.add_scalar('mINP_att', mINP_att, epoch)
    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att

def checkResult(epoch):
    print('Test Epoch: {}'.format(epoch))
    global best_acc, best_epoch
    # testing
    cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
    # save model
    if max(cmc[0], cmc_att[0]) > best_acc:  # not the real best for sysu-mm01
        best_acc = max(cmc[0], cmc_att[0])
        best_epoch = epoch
        state = {
            'net': net.state_dict(),
            'cmc': cmc_att,
            'mAP': mAP_att,
            'mINP': mINP_att,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_path + suffix + '_best.t')

    # save model
    if epoch > 10 and epoch % args.save_epoch == 0:
        state = {
            'net': net.state_dict(),
            'cmc': cmc,
            'mAP': mAP,
            'epoch': epoch,
        }
        torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

    print('POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
    print('FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
        cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
    print('Best Epoch [{}]'.format(best_epoch))

# training
print('==> Start Training...')
for epoch in range(start_epoch, 120):

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_ir_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch)

    if epoch >= 0 and epoch % 4 == 0:
        checkResult(epoch)
