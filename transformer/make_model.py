import torch
import torch.nn as nn
from resnet import ResNet, Bottleneck
import copy
from .vit_pytorch import vit_base_patch16_224_TransReID, vit_small_patch16_224_TransReID, deit_small_patch16_224_TransReID, vit_our
from losses.metric_learning import Arcface, Cosface, AMSoftmax, CircleLoss

def shuffle_unit(features, shift, group, begin=1):

    batchsize = features.size(0)
    dim = features.size(-1)
    # Shift Operation
    feature_random = torch.cat([features[:, begin-1+shift:], features[:, begin:begin-1+shift]], dim=1)
    x = feature_random
    # Patch Shuffle Operation
    try:
        x = x.view(batchsize, group, -1, dim)
    except:
        x = torch.cat([x, x[:, -2:-1, :]], dim=1)
        x = x.view(batchsize, group, -1, dim)

    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batchsize, -1, dim)

    return x

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)


class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT

        if model_name == 'resnet50':
            self.in_planes = 2048
            self.base = ResNet(last_stride=last_stride,
                               block=Bottleneck,
                               layers=[3, 4, 6, 3])
            print('using resnet50 as a backbone')
        else:
            print('unsupported backbone! but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_classes = num_classes

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None):  # label is unused if self.cos_layer == 'no'
        x = self.base(x)
        global_feat = nn.functional.avg_pool2d(x, x.shape[2:4])
        global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

        if self.neck == 'no':
            feat = global_feat
        elif self.neck == 'bnneck':
            feat = self.bottleneck(global_feat)

        if self.training:
            if self.cos_layer:
                cls_score = self.arcface(feat, label)
            else:
                cls_score = self.classifier(feat)
            return cls_score, global_feat
        else:
            if self.neck_feat == 'after':
                return feat
            else:
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory):
        super(build_transformer, self).__init__()
        last_stride = cfg.MODEL.LAST_STRIDE
        model_path = cfg.MODEL.PRETRAIN_PATH
        model_name = cfg.MODEL.NAME
        pretrain_choice = cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0
        if cfg.MODEL.SIE_VIEW:
            view_num = view_num
        else:
            view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN, sie_xishu=cfg.MODEL.SIE_COE,
                                                        camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH,
                                                        drop_rate= cfg.MODEL.DROP_OUT,
                                                        attn_drop_rate=cfg.MODEL.ATT_DROP_RATE)
        if cfg.MODEL.TRANSFORMER_TYPE == 'deit_small_patch16_224_TransReID':
            self.in_planes = 384
        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = cfg.MODEL.ID_LOSS_TYPE
        if self.ID_LOSS_TYPE == 'arcface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Arcface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'cosface':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = Cosface(self.in_planes, self.num_classes,
                                      s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'amsoftmax':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE,cfg.SOLVER.COSINE_SCALE,cfg.SOLVER.COSINE_MARGIN))
            self.classifier = AMSoftmax(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        elif self.ID_LOSS_TYPE == 'circle':
            print('using {} with s:{}, m: {}'.format(self.ID_LOSS_TYPE, cfg.SOLVER.COSINE_SCALE, cfg.SOLVER.COSINE_MARGIN))
            self.classifier = CircleLoss(self.in_planes, self.num_classes,
                                        s=cfg.SOLVER.COSINE_SCALE, m=cfg.SOLVER.COSINE_MARGIN)
        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

    def forward(self, x, label=None, cam_label= None, view_label=None):
        global_feat = self.base(x, cam_label=cam_label, view_label=view_label)

        feat = self.bottleneck(global_feat)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)

            return cls_score, global_feat  # global feature for triplet losses
        else:
            if self.neck_feat == 'after':
                # print("Test with feature after BN")
                return feat
            else:
                # print("Test with feature before BN")
                return global_feat

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


class build_transformer_local(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg, factory, rearrange, hybrid_backbone=None):
        super(build_transformer_local, self).__init__()
        model_path = cfg.MODEL.PRETRAIN_PATH
        pretrain_choice = "none"#cfg.MODEL.PRETRAIN_CHOICE
        self.cos_layer = cfg.MODEL.COS_LAYER
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        print('using Transformer_type: {} as a backbone'.format(cfg.MODEL.TRANSFORMER_TYPE))

        if cfg.MODEL.SIE_CAMERA:
            camera_num = camera_num
        else:
            camera_num = 0

        view_num = 0

        self.base = factory[cfg.MODEL.TRANSFORMER_TYPE](img_size=cfg.INPUT.SIZE_TRAIN,
                                                        hybrid_backbone=hybrid_backbone,
                                                        sie_xishu=cfg.MODEL.SIE_COE, local_feature=cfg.MODEL.JPM, camera=camera_num, view=view_num, stride_size=cfg.MODEL.STRIDE_SIZE, drop_path_rate=cfg.MODEL.DROP_PATH)

        if pretrain_choice == 'imagenet':
            self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......from {}'.format(model_path))

        block = self.base.blocks[-1]
        layer_norm = self.base.norm
        self.b1 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )
        self.b2 = nn.Sequential(
            copy.deepcopy(block),
            copy.deepcopy(layer_norm)
        )

        self.num_classes = num_classes
        self.ID_LOSS_TYPE = 'softmax'

        self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.classifier.apply(weights_init_classifier)
        self.bottleneck = nn.BatchNorm1d(self.in_planes)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)
        if self.use_JPM:
            self.classifier_1 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_1.apply(weights_init_classifier)
            self.classifier_2 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_2.apply(weights_init_classifier)
            self.classifier_3 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_3.apply(weights_init_classifier)
            self.classifier_4 = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier_4.apply(weights_init_classifier)

            self.bottleneck_1 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_1.bias.requires_grad_(False)
            self.bottleneck_1.apply(weights_init_kaiming)
            self.bottleneck_2 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_2.bias.requires_grad_(False)
            self.bottleneck_2.apply(weights_init_kaiming)
            self.bottleneck_3 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_3.bias.requires_grad_(False)
            self.bottleneck_3.apply(weights_init_kaiming)
            self.bottleneck_4 = nn.BatchNorm1d(self.in_planes)
            self.bottleneck_4.bias.requires_grad_(False)
            self.bottleneck_4.apply(weights_init_kaiming)

            self.shuffle_groups = 2
            print('using shuffle_groups size:{}'.format(self.shuffle_groups))
            self.shift_num = 5
            print('using shift_num size:{}'.format(self.shift_num))
            self.divide_length = 4
            print('using divide_length size:{}'.format(self.divide_length))
            self.rearrange = rearrange
            self.feat_dim = 5 * self.in_planes

    def forward(self, x1, x2, label=None, cam_label= None, view_label=None, modal=0):  # label is unused if self.cos_layer == 'no'

        features = self.base(x1, x2, cam_label=cam_label, view_label=view_label, modal=modal)

        # global branch
        b1_feat = self.b1(features) # [64, 129, 768]
        global_feat = b1_feat[:, 0]
        feat = self.bottleneck(global_feat)
        if self.use_JPM:
        # JPM branch
            feature_length = features.size(1) - 1
            patch_length = feature_length // self.divide_length
            token = features[:, 0:1]

            if self.rearrange:
                x = shuffle_unit(features, self.shift_num, self.shuffle_groups)
            else:
                x = features[:, 1:]
            # lf_1
            b1_local_feat = x[:, :patch_length]
            b1_local_feat = self.b2(torch.cat((token, b1_local_feat), dim=1))
            local_feat_1 = b1_local_feat[:, 0]

            # lf_2
            b2_local_feat = x[:, patch_length:patch_length*2]
            b2_local_feat = self.b2(torch.cat((token, b2_local_feat), dim=1))
            local_feat_2 = b2_local_feat[:, 0]

            # lf_3
            b3_local_feat = x[:, patch_length*2:patch_length*3]
            b3_local_feat = self.b2(torch.cat((token, b3_local_feat), dim=1))
            local_feat_3 = b3_local_feat[:, 0]

            # lf_4
            b4_local_feat = x[:, patch_length*3:patch_length*4]
            b4_local_feat = self.b2(torch.cat((token, b4_local_feat), dim=1))
            local_feat_4 = b4_local_feat[:, 0]

            local_feat_1_bn = self.bottleneck_1(local_feat_1)
            local_feat_2_bn = self.bottleneck_2(local_feat_2)
            local_feat_3_bn = self.bottleneck_3(local_feat_3)
            local_feat_4_bn = self.bottleneck_4(local_feat_4)

        if self.training:
            if self.ID_LOSS_TYPE in ('arcface', 'cosface', 'amsoftmax', 'circle'):
                cls_score = self.classifier(feat, label)
            else:
                cls_score = self.classifier(feat)
                cls_score_1 = self.classifier_1(local_feat_1_bn)
                cls_score_2 = self.classifier_2(local_feat_2_bn)
                cls_score_3 = self.classifier_3(local_feat_3_bn)
                cls_score_4 = self.classifier_4(local_feat_4_bn)
            return [cls_score, cls_score_1, cls_score_2, cls_score_3,
                        cls_score_4
                        ], [global_feat, local_feat_1, local_feat_2, local_feat_3,
                            local_feat_4]  # global feature for triplet losses
        else:
            if self.neck_feat == 'after':
                return torch.cat(
                    [feat, local_feat_1_bn / 4, local_feat_2_bn / 4, local_feat_3_bn / 4, local_feat_4_bn / 4], dim=1)
            else:
                return torch.cat(
                    [global_feat, local_feat_1 / 4, local_feat_2 / 4, local_feat_3 / 4, local_feat_4 / 4], dim=1)

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        for i in param_dict:
            self.state_dict()[i.replace('module.', '')].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)



__factory_T_type = {
    'vit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'deit_base_patch16_224_TransReID': vit_base_patch16_224_TransReID,
    'vit_small_patch16_224_TransReID': vit_small_patch16_224_TransReID,
    'deit_small_patch16_224_TransReID': deit_small_patch16_224_TransReID,
    'vit_our': vit_our
}

def make_cfg():
    cfg = type('test', (), {})()
    cfg.MODEL = type('test', (), {})()
    cfg.INPUT = type('test', (), {})()
    cfg.DATASETS = type('test', (), {})()
    cfg.DATALOADER = type('test', (), {})()
    cfg.SOLVER = type('test', (), {})()
    cfg.TEST = type('test', (), {})()

    cfg.MODEL.PRETRAIN_CHOICE= 'imagenet'
    cfg.MODEL.PRETRAIN_PATH= '/home/mahdi/.cache/torch/checkpoints/jx_vit_base_p16_224-80ecf9dd.pth'
    cfg.MODEL.METRIC_LOSS_TYPE= 'triplet'
    cfg.MODEL.IF_LABELSMOOTH= 'off'
    cfg.MODEL.IF_WITH_CENTER= 'no'
    cfg.MODEL.NAME= 'transformer'
    cfg.MODEL.NO_MARGIN= True
    cfg.MODEL.DEVICE_ID= ('5')
    cfg.MODEL.TRANSFORMER_TYPE= 'vit_our'
    cfg.MODEL.STRIDE_SIZE= [12, 12]
    cfg.MODEL.SIE_CAMERA= True
    cfg.MODEL.SIE_COE= 3.0
    cfg.MODEL.JPM= True
    cfg.MODEL.RE_ARRANGE= True
    cfg.MODEL.COS_LAYER = False
    cfg.MODEL.NECK = 'bnneck'
    cfg.TEST.NECK_FEAT = 'before'
    cfg.MODEL.DROP_PATH = 0.1


    cfg.INPUT.SIZE_TRAIN= [288, 144]
    cfg.INPUT.SIZE_TEST= [288, 144]
    cfg.INPUT.PROB= 0.5 # random horizontal flip
    cfg.INPUT.RE_PROB= 0.5 # random erasing
    cfg.INPUT.PADDING= 10
    cfg.INPUT.PIXEL_MEAN= [0.5, 0.5, 0.5]
    cfg.INPUT.PIXEL_STD= [0.5, 0.5, 0.5]


    cfg.DATASETS.NAMES = ('market1501')
    cfg.DATASETS.ROOT_DIR = ('./data')


    cfg.DATALOADER.SAMPLER = 'softmax_triplet'
    cfg.DATALOADER.NUM_INSTANCE = 4
    cfg.DATALOADER.NUM_WORKERS = 0


    cfg.SOLVER.OPTIMIZER_NAME = 'SGD'
    cfg.SOLVER.MAX_EPOCHS = 120
    cfg.SOLVER.BASE_LR = 0.008
    cfg.SOLVER.IMS_PER_BATCH = 16
    cfg.SOLVER.WARMUP_METHOD = 'linear'
    cfg.SOLVER.LARGE_FC_LR = False
    cfg.SOLVER.CHECKPOINT_PERIOD = 120
    cfg.SOLVER.LOG_PERIOD = 50
    cfg.SOLVER.EVAL_PERIOD = 120
    cfg.SOLVER.WEIGHT_DECAY =  1e-4
    cfg.SOLVER.WEIGHT_DECAY_BIAS = 1e-4
    cfg.SOLVER.BIAS_LR_FACTOR = 2
    cfg.SOLVER.MOMENTUM = 0.9
    cfg.SOLVER.CENTER_LR = 0.5

    cfg.TEST.EVAL = True
    cfg.TEST.IMS_PER_BATCH = 256
    cfg.TEST.RE_RANKING = False
    cfg.TEST.WEIGHT = ''
    cfg.TEST.NECK_FEAT = 'before'
    cfg.TEST.FEAT_NORM = 'yes'
    return cfg


def make_model(num_class, camera_num, view_num, hybrid_backbone=None):
    cfg = make_cfg()
    model = build_transformer_local(num_class, camera_num, view_num, cfg, __factory_T_type, rearrange=True, hybrid_backbone=hybrid_backbone)
    print('===========building transformer with JPM module ===========')
    return model


def make_optimizer(model, center_criterion):
    cfg = make_cfg()
    params = []
    num = 0
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        if cfg.SOLVER.LARGE_FC_LR:
            if "classifier" in key or "arcface" in key:
                lr = cfg.SOLVER.BASE_LR * 2
                print('Using two times learning rate for fc ')

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        num = num + value.numel()
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    elif cfg.SOLVER.OPTIMIZER_NAME == 'AdamW':
        optimizer = torch.optim.AdamW(params, lr=cfg.SOLVER.BASE_LR, weight_decay=cfg.SOLVER.WEIGHT_DECAY)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)
    optimizer_center = torch.optim.SGD(center_criterion.parameters(), lr=cfg.SOLVER.CENTER_LR)

    return optimizer, optimizer_center
