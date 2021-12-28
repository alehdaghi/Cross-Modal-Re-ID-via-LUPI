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
from model_cmalign import embed_net
from sup_con_loss import SupConLoss
from utils import *
from loss import *
from tensorboardX import SummaryWriter
from cm_align import CMAlign

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
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='agw', type=str,
                    metavar='m', help='method type: base or agw')
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
parser.add_argument('--mode', default='all', type=str, help='all or indoor')
parser.add_argument('--use_gray', dest='use_gray', help='use gray as 3rd modality', action='store_true')
parser.add_argument('--separate_batch_norm', dest='separate_batch_norm', help='separate batch norm layers only in first layers',
                    action='store_true')
parser.add_argument('--cont', dest='cont_loss', help='use Contrastive Loss', action='store_true')
parser.set_defaults(use_gray=False)
parser.set_defaults(separate_batch_norm=False)
parser.set_defaults(cont_loss=False)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '../Datasets/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    if args.uni == 0:
        test_mode = [1, 2]  # thermal to visible
    else:
        test_mode = [args.uni, args.uni]  # thermal to visible
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
suffix = suffix + '_cmalign_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)


if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if args.use_gray:
    suffix = suffix + '_gray'

if args.cont_loss:
    suffix = suffix + '_cont'


if args.separate_batch_norm:
    suffix = suffix + '_sepBatch'

if args.arch == 'resnet18':
    suffix = suffix + '_arch18'



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
    trainset = SYSUData(data_path, transform=transform_train, gray=(args.use_gray or args.uni == 3))
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_ir_label)

    # testing set
    if args.uni == 1:
        args.mode = 'Vis'
    elif args.uni == 2:
        args.mode = 'Ir'
    elif args.uni == 3:
        args.mode = 'Gray'
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_ir_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, gall_cam, transform=transform_test, img_size=(args.img_w, args.img_h), colorToGray= args.uni == 3)
queryset = TestData(query_img, query_label, query_cam, transform=transform_test, img_size=(args.img_w, args.img_h), colorToGray= args.uni == 3)

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
if args.method =='base':
    net = embed_net(n_class, no_local= 'off', gm_pool =  'off', arch=args.arch, separate_batch_norm=args.separate_batch_norm)
else:
    net = embed_net(n_class, no_local= 'on', gm_pool = 'on', arch=args.arch, separate_batch_norm=args.separate_batch_norm)

net.cmalign = CMAlign(args.batch_size, args.num_pos, use_gray=args.use_gray)
net.to(device)
cudnn.benchmark = True
print(net.count_params())
pool_dim = 2048
if args.arch == 'resnet18':
    pool_dim = 512
if len(args.resume) > 0:
    model_path = args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define losses function
criterion_id = nn.CrossEntropyLoss()
if args.method == 'agw':
    criterion_tri = TripletLoss_WRT()
else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri = OriTripletLoss(batch_size=loader_batch, margin=args.margin)

cross_triplet_creiteron = TripletLoss(0.3, 'euclidean')
reconst_loss = nn.MSELoss()
hetro_loss = HetroCenterLoss()
hctriplet = HcTripletLoss(margin=0.8)


criterion_id.to(device)
criterion_tri.to(device)
cross_triplet_creiteron.margin_loss.to(device)
reconst_loss.to(device)

criterion_contrastive = SupConLoss()

if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))
    ids = set(map(id, net.parameters()))
    params = filter(lambda p: id(p) in ids, net.parameters())
    base_params = filter(lambda p: id(p) not in ignored_params, params)

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)

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
    KL_loss = AverageMeter()
    A_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()

    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, input3, label1, label2, _,cam1, cam2) in enumerate(trainloader):

        bs = label1.shape[0]
        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())
        if args.use_gray or args.uni == 3:
            labels = torch.cat((label1, label2, label1), 0)
            input3 = Variable(input3.cuda())
        else:
            input3 = None
            labels = torch.cat((label1, label2), 0)

        if args.uni == 1 or args.uni == 3:
            labels = label1
        elif args.uni == 2:
            labels = label2

        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)

        feat, out0, align_outs = net(input1, input2, x3=input3, modal=args.uni)

        loss_color2gray = torch.tensor(0.0, requires_grad=True, device=device)
        if args.use_gray:
            color_feat, thermal_feat, gray_feat = torch.split(feat, label1.shape[0])
            color_label, thermal_label, gray_label = torch.split(labels, label1.shape[0])
            loss_tri_color = cross_triplet_creiteron(color_feat, thermal_feat, gray_feat,
                                                     color_label, thermal_label, gray_label)
            loss_tri_thermal = cross_triplet_creiteron(thermal_feat, gray_feat, color_feat,
                                                       thermal_label, gray_label, color_label)
            loss_tri_gray = cross_triplet_creiteron(gray_feat, color_feat, thermal_feat,
                                                    gray_label, color_label, thermal_label)
            loss_tri = (loss_tri_color + loss_tri_thermal + loss_tri_gray) / 3
            loss_color2gray = reconst_loss(color_feat, gray_feat)

        else:
            if args.uni != 0:
                color_feat, thermal_feat = feat, feat
                color_label, thermal_label = labels, labels
            else:
                color_feat, thermal_feat = torch.split(feat, bs)
                color_label, thermal_label = torch.split(labels, bs)
            loss_tri_color = cross_triplet_creiteron(color_feat, thermal_feat, thermal_feat,
                                                     color_label, thermal_label, thermal_label)
            loss_tri_thermal = cross_triplet_creiteron(thermal_feat, color_feat, color_feat,
                                                       thermal_label, color_label, color_label)
            loss_tri = (loss_tri_color + loss_tri_thermal) / 2

        loss_id = criterion_id(out0, labels)
        loss = torch.tensor(0.0, requires_grad=True, device=device)
        if align_outs is not None:
            loss_KL = F.kl_div(F.log_softmax(out0, dim=1), F.softmax(align_outs['cls_ic_layer4'], dim=1), reduction='batchmean')
            loss_id += criterion_id(align_outs['cls_ic_layer4'], labels)
            loss_id /= 2
            tri_loss.update(loss_tri.item(), 2 * input1.size(0))
            KL_loss.update(loss_KL.item(), 2 * input1.size(0))
            A_loss.update(align_outs['loss_dt'].item(), 2 * input1.size(0))
            loss = align_outs['loss_dt'] + loss_KL
        #loss_tri, batch_acc = criterion_tri(feat, labels)
        #loss_center = hetro_loss(color_feat, thermal_feat, color_label, thermal_label)
        #l1, _ = hctriplet(feat, labels)


        #correct += (batch_acc / 2)
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item() / 2)

        if args.cont_loss:
            feat = torch.cat([F.normalize(color_feat, dim=1).unsqueeze(1), F.normalize(thermal_feat, dim=1).unsqueeze(1)], dim=1)
            loss_cont = criterion_contrastive(feat, labels[:bs])
            loss += loss_cont + loss_id
        else:
            loss += loss_id + loss_tri + loss_color2gray #+ loss_center

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))

        gray_loss.update(loss_color2gray.item(), 2 * input1.size(0))

        total += labels.size(0)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('E: [{}][{}/{}] '
                  'T: {now} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'il: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'Tl: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'Gl: {gray_loss.val:.4f} ({gray_loss.avg:.4f}) '
                  'KLl: {KL_loss.val:.4f} ({KL_loss.avg:.4f}) '
                  'Al: {A_loss.val:.4f} ({A_loss.avg:.4f}) '
                  'Acc: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, now=time_now(), batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss,
                gray_loss=gray_loss, KL_loss=KL_loss, A_loss=A_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('gray_loss', gray_loss.avg, epoch)
    writer.add_scalar('KL_loss', KL_loss.avg, epoch)
    writer.add_scalar('A_loss', A_loss.avg, epoch)
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


# training
print('==> Start Training...')
for epoch in range(start_epoch, 82):

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
        print('Test Epoch: {}'.format(epoch))

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
