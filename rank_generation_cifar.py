import torch
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import os
import argparse
from torch.utils.data import Dataset, DataLoader
import data.imagenet as imagenet
from models import *
from utils import progress_bar
import numpy as np


def get_same_index(target, label):
    label_indices = []

    for i in range(len(target)):
        if target[i] == label:
            label_indices.append(i)

    return label_indices


parser = argparse.ArgumentParser(description='Rank extraction')

parser.add_argument(
    '--data_dir',
    type=str,
    default='./data',
    help='dataset path')
parser.add_argument(
    '--dataset',
    type=str,
    default='cifar10',
    choices=('cifar10', 'imagenet'),
    help='dataset')
parser.add_argument(
    '--job_dir',
    type=str,
    default='result/tmp',
    help='The directory where the summaries will be stored.')
parser.add_argument(
    '--arch',
    type=str,
    default='vgg_16_bn',  # 'vgg_16_bn', # 'resnet_56'
    choices=('resnet_50', 'vgg_16_bn', 'resnet_56', 'resnet_110', 'densenet_40', 'googlenet'),
    help='The architecture to prune')
parser.add_argument(
    '--resume',
    type=str,
    default="./pretrained_models/VGG_16/vgg_16_bn.pt", # "./pretrained_models/ResNet56/resnet_56.pt" # "./pretrained_models/VGG_16/vgg_16_bn.pt"
    help='load the model from the specified checkpoint')
parser.add_argument(
    '--train_batch_size',
    type=int,
    default=128,
    help='Batch size for training.')
parser.add_argument(
    '--eval_batch_size',
    type=int,
    default=100,
    help='Batch size for validation.')
parser.add_argument(
    '--start_idx',
    type=int,
    default=0,
    help='The index of conv to start extract rank.')
parser.add_argument(
    '--gpu',
    type=str,
    default='0',
    help='Select gpu to use')
parser.add_argument(
    '--adjust_ckpt',
    action='store_true',
    help='adjust ckpt from pruned checkpoint')
parser.add_argument(
    '--compress_rate',
    type=str,
    default=None,
    help='compress rate of each conv')

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if args.compress_rate:
    import re

    cprate_str = args.compress_rate
    cprate_str_list = cprate_str.split('+')
    pat_cprate = re.compile(r'\d+\.\d*')
    pat_num = re.compile(r'\*\d+')
    cprate = []
    for x in cprate_str_list:
        num = 1
        find_num = re.findall(pat_num, x)
        if find_num:
            assert len(find_num) == 1
            num = int(find_num[0].replace('*', ''))
        find_cprate = re.findall(pat_cprate, x)
        assert len(find_cprate) == 1
        cprate += [float(find_cprate[0])] * num

    compress_rate = cprate
else:
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
    compress_rate = default_cprate[args.arch]

# Model
print('==> Building model..')
print(compress_rate)

for run in range(0, 10):

    # Set params
    print("run is ", run)
    label_class = run  # runs for 1000 classes

    print('==> Preparing data..')

    pin_memory = True

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = CIFAR10(root=args.data_dir, train=True, download=True, transform=transform_train)

    train_indices = get_same_index(trainset.targets, label_class)

    trainloader = DataLoader(
        trainset,
        #                 batch_size=args.train_batch_size,
        batch_size=128,
        #                 shuffle=True, # commented this
        num_workers=2,
        pin_memory=pin_memory,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(train_indices)  # I added this
    )

    testset = CIFAR10(root=args.data_dir, train=False, download=True, transform=transform_test)
    testloader = DataLoader(
        testset,
        #             batch_size=args.eval_batch_size,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    net = eval(args.arch)(compress_rate=compress_rate)
    net = net.to(device)

    if len(args.gpu) > 1 and torch.cuda.is_available():
        device_id = []
        for i in range((len(args.gpu) + 1) // 2):
            device_id.append(i)
        net = torch.nn.DataParallel(net, device_ids=device_id)

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        checkpoint = torch.load(args.resume, map_location='cuda:' + args.gpu)
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        # if args.arch == 'resnet_56':
        #     print("I'm in")
        #     for k, v in checkpoint.items():
        #         new_state_dict[k.replace('module.', '')] = v
        # else:
            # if args.adjust_ckpt:
        for k, v in checkpoint['state_dict'].items():
            new_state_dict[k.replace('module.', '')] = v
        # print(new_state_dict)
        # print()
        net.load_state_dict(new_state_dict)

    criterion = nn.CrossEntropyLoss()
    feature_result = torch.tensor(0.)
    total = torch.tensor(0.)


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

                progress_bar(batch_idx, limit, 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                             % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))  # '''


    if args.arch == 'vgg_16_bn':

        if len(args.gpu) > 1:
            relucfg = net.module.relucfg
        else:
            relucfg = net.relucfg

        for i, cov_id in enumerate(relucfg):
            cov_layer = net.features[cov_id]
            handler = cov_layer.register_forward_hook(get_feature_hook)
            test()
            handler.remove()

            if not os.path.isdir('./rank_conv/' + args.arch):
                os.mkdir('./rank_conv/' + args.arch)
            np.save('./rank_conv/' + args.arch + '/rank_conv%d' % (i + 1) + '_Class_' + str(label_class) + '.npy',
                    feature_result.numpy())

            feature_result = torch.tensor(0.)
            total = torch.tensor(0.)

    elif args.arch == 'resnet_56':

        cov_layer = eval('net.relu')
        handler = cov_layer.register_forward_hook(get_feature_hook)
        test()
        handler.remove()

        if not os.path.isdir('./rank_conv/' + args.arch):
            os.mkdir('./rank_conv/' + args.arch)
        np.save('./rank_conv/' + args.arch + '/rank_conv%d' % (1) + '_Class_' + str(label_class) + '.npy',
                feature_result.numpy())
        feature_result = torch.tensor(0.)
        total = torch.tensor(0.)

        # ResNet56 per block
        cnt = 1
        for i in range(3):
            block = eval('net.layer%d' % (i + 1))
            for j in range(9):
                cov_layer = block[j].relu1
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                np.save('./rank_conv/' + args.arch + '/rank_conv%d' % (cnt + 1) + '_Class_' + str(label_class) + '.npy',
                        feature_result.numpy())
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)

                cov_layer = block[j].relu2
                handler = cov_layer.register_forward_hook(get_feature_hook)
                test()
                handler.remove()
                np.save('./rank_conv/' + args.arch + '/rank_conv%d' % (cnt + 1) + '_Class_' + str(label_class) + '.npy',
                        feature_result.numpy())
                cnt += 1
                feature_result = torch.tensor(0.)
                total = torch.tensor(0.)
