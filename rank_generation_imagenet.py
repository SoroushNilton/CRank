from __future__ import absolute_import
import torch
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

import sys
import time
import logging
import datetime
from pathlib import Path
import numpy as np

data_root = "/mnt/Research/imagenetData/imagenetdata"  #"PATH OF DATA"  # e.g. '/mnt/imagenetdata/'
model_path = "./pretrained_models/ResNet50/resnet50-19c8e357.pth"


# Return only images of certain class (eg. airplanes = class 0)
def get_same_index(target, label):
    label_indices = []

    for i in range(len(target)):
        if target[i] == label:
            label_indices.append(i)

    return label_indices


class Data:
    def __init__(self, args, is_evaluate=False):
        pin_memory = False
        #         if args.gpu is not None:
        pin_memory = True

        scale_size = 224

        traindir = os.path.join(data_root, 'ILSVRC2012_img_train')
        valdir = os.path.join(data_root, 'val')
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if not is_evaluate:
            trainset = datasets.ImageFolder(
                traindir,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.Resize(scale_size),
                    transforms.ToTensor(),
                    normalize,
                ]))

            # Get indices of label_class
            train_indices = get_same_index(trainset.targets, label_class)  # added this

            self.loader_train = DataLoader(
                trainset,
                #                 batch_size=args.train_batch_size,
                batch_size=10,
                #                 shuffle=True, # commented this
                num_workers=2,
                pin_memory=pin_memory,
                sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices)  # I added this
            )

        testset = datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(scale_size),
                transforms.ToTensor(),
                normalize,
            ]))

        self.loader_test = DataLoader(
            testset,
            #             batch_size=args.eval_batch_size,
            batch_size=6,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )


import torch.nn as nn

norm_mean, norm_var = 1.0, 0.1


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, cp_rate=None, tmp_name=None):
        super(ResBottleneck, self).__init__()
        if cp_rate is None:
            cp_rate = []
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.conv1.cp_rate = cp_rate[0]
        self.conv1.tmp_name = tmp_name
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2.cp_rate = cp_rate[1]
        self.conv2.tmp_name = tmp_name
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.conv3.cp_rate = cp_rate[2]
        self.conv3.tmp_name = tmp_name

        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu3(out)

        return out


class Downsample(nn.Module):
    def __init__(self, downsample):
        super(Downsample, self).__init__()
        self.downsample = downsample

    def forward(self, x):
        out = self.downsample(x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, covcfg=None, compress_rate=None):
        self.inplanes = 64
        super(ResNet, self).__init__()

        self.covcfg = covcfg
        self.compress_rate = compress_rate
        self.num_blocks = num_blocks

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.conv1.cp_rate = compress_rate[0]
        self.conv1.tmp_name = 'conv1'
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1,
                                       cp_rate=compress_rate[1:3 * num_blocks[0] + 2],
                                       tmp_name='layer1')
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2,
                                       cp_rate=compress_rate[
                                               3 * num_blocks[0] + 2:3 * num_blocks[0] + 3 * num_blocks[1] + 3],
                                       tmp_name='layer2')
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2,
                                       cp_rate=compress_rate[
                                               3 * num_blocks[0] + 3 * num_blocks[1] + 3:3 * num_blocks[0] + 3 *
                                                                                         num_blocks[1] + 3 * num_blocks[
                                                                                             2] + 4],
                                       tmp_name='layer3')
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2,
                                       cp_rate=compress_rate[
                                               3 * num_blocks[0] + 3 * num_blocks[1] + 3 * num_blocks[2] + 4:],
                                       tmp_name='layer4')

        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride, cp_rate, tmp_name):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            conv_short = nn.Conv2d(self.inplanes, planes * block.expansion,
                                   kernel_size=1, stride=stride, bias=False)
            conv_short.cp_rate = cp_rate[0]
            conv_short.tmp_name = tmp_name + '_shortcut'
            downsample = nn.Sequential(
                conv_short,
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, cp_rate=cp_rate[1:4],
                            tmp_name=tmp_name + '_block' + str(1)))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, cp_rate=cp_rate[3 * i + 1:3 * i + 4],
                                tmp_name=tmp_name + '_block' + str(i + 1)))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)

        # 256 x 56 x 56
        x = self.layer2(x)

        # 512 x 28 x 28
        x = self.layer3(x)

        # 1024 x 14 x 14
        x = self.layer4(x)

        # 2048 x 7 x 7
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet_50(compress_rate=None):
    cov_cfg = [(3 * i + 3) for i in range(3 * 3 + 1 + 4 * 3 + 1 + 6 * 3 + 1 + 3 * 3 + 1 + 1)]
    model = ResNet(ResBottleneck, [3, 4, 6, 3], covcfg=cov_cfg, compress_rate=compress_rate)
    return model


def get_logger(file_path):
    """ Make python logger """
    # [!] Since tensorboardX use default logger (e.g. logging.info()), we should use custom logger
    logger = logging.getLogger('kd')
    log_format = '%(asctime)s | %(message)s'
    formatter = logging.Formatter(log_format, datefmt='%m/%d %I:%M:%S %p')
    file_handler = logging.FileHandler(file_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO)

    return logger


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class checkpoint():
    def __init__(self, args):
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
        today = datetime.date.today()

        self.args = args
        self.job_dir = Path(args.job_dir)
        self.run_dir = self.job_dir / 'run'
        print(args.job_dir)

        def _make_dir(path):
            if not os.path.exists(path): os.makedirs(path)

        _make_dir(self.job_dir)
        _make_dir(self.run_dir)

        config_dir = self.job_dir / 'config.txt'
        with open(config_dir, 'w') as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')


def print_params(config, prtf=print):
    prtf("")
    prtf("Parameters:")
    for attr, value in sorted(config.items()):
        prtf("{}={}".format(attr.upper(), value))
    prtf("")


# _, term_width = os.popen('stty size', 'r').read().split()
term_width = int(80)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time


def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH * current / total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width - int(TOTAL_BAR_LENGTH) - len(msg) - 3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width - int(TOTAL_BAR_LENGTH / 2) + 2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current + 1, total))

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds * 1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for run in range(0, 1000):

    # Set params
    print("run is ", run)
    batch_size = 10
    label_class = run  # runs for 1000 classes

    # ================= Data Block
    # Data
    print('==> Preparing data..')

    data_tmp = Data('imagenet')
    trainloader = data_tmp.loader_train
    testloader = data_tmp.loader_test

    default_cprate = {
        'vgg_16_bn': [0.7] * 7 + [0.1] * 6,
        'densenet_40': [0.0] + [0.1] * 6 + [0.7] * 6 + [0.0] + [0.1] * 6 + [0.7] * 6 + [0.0] + [0.1] * 6 + [0.7] * 5 + [
            0.0],
        'googlenet': [0.10] + [0.7] + [0.5] + [0.8] * 4 + [0.5] + [0.6] * 2,
        'resnet_50': [0.2] + [0.8] * 10 + [0.8] * 13 + [0.55] * 19 + [0.45] * 10,
        'resnet_56': [0.1] + [0.60] * 35 + [0.0] * 2 + [0.6] * 6 + [0.4] * 3 + [0.1] + [0.4] + [0.1] + [0.4] + [0.1] + [
            0.4] + [0.1] + [0.4],
        'resnet_110': [0.1] + [0.40] * 36 + [0.40] * 36 + [0.4] * 36
    }
    compress_rate = default_cprate['resnet_50']

    # ================ Model Block
    # Model
    print('==> Building model..')
    # print(compress_rate)
    net = eval('resnet_50')(compress_rate=compress_rate)
    net = net.to(device)

    # loading pre trained model
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(model_path, map_location='cuda:' + '0')
    from collections import OrderedDict

    new_state_dict = OrderedDict()
    # if args.adjust_ckpt:
    for k, v in checkpoint.items():
        new_state_dict[k.replace('module.', '')] = v
    # else:
    #     for k, v in checkpoint['state_dict'].items():
    #         new_state_dict[k.replace('module.', '')] = v
    net.load_state_dict(new_state_dict)

    # ================ loss block
    criterion = nn.CrossEntropyLoss()
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)


    # =============== Feature Hook

    def get_feature_hook(self, input, output):
        global feature_result
        global entropy
        global total
        a = output.shape[0]
        b = output.shape[1]
        c = torch.tensor([torch.matrix_rank(output[i, j, :, :]).item() for i in range(a) for j in range(b)])

        c = c.view(a, -1).float()
        c = c.sum(0)
        feature_result = feature_result * total + c
        total = total + a
        feature_result = feature_result / total


    # ================= Test Block

    def test():
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        limit = 5

        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(trainloader):
                if batch_idx >= limit:  # use the first 6 batches to estimate the rank.
                    break
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()


    # Last Block
    cov_layer = eval('net.maxpool')
    handler = cov_layer.register_forward_hook(get_feature_hook)
    test()
    handler.remove()

    if not os.path.isdir('./rank_conv/' + 'resnet_50'):
        os.mkdir('./rank_conv/resnet_50')

    np.save('./rank_conv/' + 'resnet_50' + '/rank_conv%d' % (1) +
            '_Class_' + str(label_class) + '.npy', feature_result.numpy())
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)

    # ResNet50 per bottleneck
    cnt = 1
    for i in range(4):
        block = eval('net.layer%d' % (i + 1))
        for j in range(net.num_blocks[i]):
            cov_layer = block[j].relu1
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            np.save('rank_conv/' + 'resnet_50' + '/rank_conv%d' % (cnt + 1) +
                    '_Class_' + str(label_class) + '.npy', feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu2
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            np.save('./rank_conv/' + 'resnet_50' + '/rank_conv%d' % (cnt + 1) +
                    '_Class_' + str(label_class) + '.npy',
                    feature_result.numpy())
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

            cov_layer = block[j].relu3
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()
            if j == 0:
                np.save('./rank_conv/' + 'resnet_50' + '/rank_conv%d' % (cnt + 1) +
                        '_Class_' + str(label_class) + '.npy',
                        feature_result.numpy())  # shortcut conv
                cnt += 1
            np.save('./rank_conv/' + 'resnet_50' + '/rank_conv%d' % (cnt + 1) +
                    '_Class_' + str(label_class) + '.npy',
                    feature_result.numpy())  # conv3
            cnt += 1
            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

    print('============= class ==== ', label_class, '=== Finished! =============')
