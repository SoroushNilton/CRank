import os
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

# # Return only images of certain class (eg. airplanes = class 0)
# def get_same_index(target, label):
#     label_indices = []
#
#     for i in range(len(target)):
#         if target[i] == label:
#             label_indices.append(i)
#
#     return label_indices


class Data:
    def __init__(self, args, is_evaluate=False):
        pin_memory = False
        if args.gpu is not None:
            pin_memory = True

        scale_size = 224

        traindir = os.path.join(args.data_dir, 'ILSVRC2012_img_train')
        valdir = os.path.join(args.data_dir, 'val')
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

            # # Get indices of label_class
            # train_indices = get_same_index(trainset.targets, label_class)


            self.loader_train = DataLoader(
                trainset,
                batch_size=args.train_batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=pin_memory)

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
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True)
