import random
import numpy as np
import os

from logging import getLogger
from PIL import ImageFilter
from PIL import Image
import torchvision.transforms as transforms
import torch
from torchvision.transforms import functional as F

logger = getLogger()

class MultiCropDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            root,
            size_crops,
            nmb_crops,
            min_scale_crops,
            max_scale_crops,
            size_dataset=-1,
            return_index=False,):
        r"""
        Args:
            root: Location of the dataset folder, usually it is /unlabeled
            transform: the transform you want to applied to the images.
        """

        # get file path list
        self.image_dir = root
        self.filelist = os.listdir(self.image_dir)
        # get rid of the corrupted images
        problem_img = ['15460.PNG','151438.PNG', '158432.PNG']
        for p in problem_img:
            if p in self.filelist:
                self.filelist.remove(p)

        # get file length
        if size_dataset >= 0:
            self.filelist = self.filelist[:size_dataset]
        self.num_images = len(self.filelist)

        assert len(size_crops) == len(nmb_crops)
        assert len(min_scale_crops) == len(nmb_crops)
        assert len(max_scale_crops) == len(nmb_crops)

        # transformation
        color_transform = [get_color_distortion(), PILRandomGaussianBlur()]
        mean = [0.4917, 0.4694, 0.4148] # [0.485, 0.456, 0.406] for imagenet
        std = [0.2278, 0.2240, 0.2280] # [0.228, 0.224, 0.225] for imagenet
        trans = []
        for i in range(len(size_crops)):
            randomresizedcrop = transforms.RandomResizedCrop(
                size_crops[i],
                scale=(min_scale_crops[i], max_scale_crops[i]),
            )
            trans.extend([transforms.Compose([
                randomresizedcrop,
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Compose(color_transform),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)])
            ] * nmb_crops[i])
        self.trans = trans


    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        # the idx of unlabeled image is not consecutive
        with open(os.path.join(self.image_dir, self.filelist[idx]), 'rb') as f:
            img = Image.open(f).convert('RGB')
            multi_crops = list(map(lambda trans: trans(img), self.trans))
            return multi_crops








class PILRandomGaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image. Take the radius and probability of
    application as the parameter.
    This transform was used in SimCLR - https://arxiv.org/abs/2002.05709
    """

    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = np.random.rand() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


def get_color_distortion(s=1.0):
    # s is the strength of color distortion.
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
    return color_distort