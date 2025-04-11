# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

"""
MIRFlickr25K
IAPR-TC12
"""

from logging import getLogger

from PIL import ImageFilter, Image
import PIL
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.utils.data as data
import scipy.io as sio
import os
import h5py
import torch
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image
from utils.config import args

# device = torch.device('cuda:1')


logger = getLogger()

class Sampler():
    def __init__(self, root, paths):
        self.root = root
        if isinstance(paths, np.ndarray):
            if len(paths.shape) == 1 or paths.shape[0] == 1 or paths.shape[1] == 1:
                paths = paths.reshape([-1]).tolist()
        self.paths = paths

    def __getitem__(self, item):
        path = self.paths[item]
        if isinstance(path, np.ndarray):
            if len(path.shape) >= 2:
                return Image.fromarray(path, mode='RGB')
            else:
                path = path[0]
        return Image.open(os.path.join(self.root, path))

    def __len__(self):
        return len(self.paths)

def text_transform(text):
    return text

class CMDataset(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(
        self,
        data_name,
        return_index=False,
        partition='train'
    ):
        self.NUSWIDE=None
        self.iapr=None
        self.mscoco=None
        self.data_name = data_name
        self.partition = partition
        training = 'train' in partition.lower()
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans = []

        if training:
            trans.extend([transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)])
                ])
        else:
            trans.extend([transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ])

        self.trans = trans
        self.return_index = return_index
        self.open_data()


    def open_data(self):
        if self.data_name.lower() == 'mirflickr25k':
            data = MIRFlickr25K(self.partition)
        elif self.data_name.lower()=="iapr":
            data=IAPR_TC12(self.partition)
            self.iapr=True

        if len(data) == 3:
            (self.imgs, self.texts, self.labels) = data
            self.imgs = self.imgs
        else:
            (self.imgs, self.texts, self.labels, root) = data
            self.imgs = Sampler(root, self.imgs)
        self.length = self.labels.shape[0]
        self.text_dim = self.texts.shape[1]


    def __getitem__(self, index,partition="train"):
        image = self.imgs[index]
        text = self.texts[index]
        if self.NUSWIDE or self.iapr or self.mscoco:
            image = image.convert("RGB")

        if isinstance(self.imgs, Sampler):
            multi_crops = list(map(lambda trans: trans(image), self.trans))
            text = list(map(lambda trans: trans(text), [text_transform] * len(self.trans)))
        else:
            multi_crops = [image]
            text = [text]

        label = self.labels[index]

        if self.return_index:
            return index, multi_crops, text, label
        return multi_crops, text, label
        # return multi_crops, text, index

    def __len__(self):
        return self.length

def MIRFlickr25K(partition):
    imgs = sio.loadmat('./data/FAll/mirflickr25k-fall.mat')['FAll']
    root = '/home/yifan_wang/UCCH/mirflickr25k/mirflickr/'
    tags = sio.loadmat('./data/YAll/mirflickr25k-yall.mat')['YAll']
    labels = sio.loadmat('./data/LAll/mirflickr25k-lall.mat')['LAll']

    inx = np.arange(imgs.shape[0])
    np.random.shuffle(inx)
    imgs, tags, labels = imgs[inx], tags[inx], labels[inx]
    test_size = 2000
    if 'test' in partition.lower():
        imgs, tags, labels = imgs[-test_size::], tags[-test_size::], labels[-test_size::]
    else:
        imgs, tags, labels = imgs[0: -test_size], tags[0: -test_size], labels[0: -test_size]


    return imgs, tags, labels,root

def IAPR_TC12(partition):
    imgs=np.load("/home/yifan_wang/UCCH/iapr/img.npy")
    texts=np.load("/home/yifan_wang/UCCH/iapr/texts.npy")
    labels=np.load("/home/yifan_wang/UCCH/iapr/labels.npy")
    root = '/home/yifan_wang/UCCH/IAPR-TC12/final_data'

    inx = np.arange(len(imgs))
    np.random.shuffle(inx)
    imgs, texts, labels = imgs[inx], texts[inx], labels[inx]
    # 数据划分
    test_size = 2000

    if 'test' in partition.lower():
        imgs, texts, labels = imgs[:test_size], texts[:test_size], labels[:test_size]
    else:
        imgs, texts, labels = imgs[test_size :], texts[test_size :], labels[test_size :]

    return imgs, texts, labels, root
