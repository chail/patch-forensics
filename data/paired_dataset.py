import os.path
import torch.utils.data as data
from .dataset_util import make_dataset
from PIL import Image
import numpy as np
import torch
from . import transforms
import random

class PairedDataset(data.Dataset):
    """A dataset class for paired images
    e.g. corresponding real and manipulated images
    """

    def __init__(self, opt, im_path_real, im_path_fake, is_val=False):
        """Initialize this dataset class.

        Parameters:
            opt -- experiment options
            im_path_real -- path to folder of real images
            im_path_fake -- path to folder of fake images
            is_val -- is this training or validation? used to determine
            transform
        """
        super().__init__()
        self.dir_real = im_path_real
        self.dir_fake = im_path_fake

        # if pairs are named in the same order 
        # e.g. real/train/face1.png, real/train/face2.png ...
        #      fake/train/face1.png, fake/train/face2.png ...
        # then this will align them in a batch unless
        # --no_serial_batches is specified
        self.real_paths = sorted(make_dataset(self.dir_real,
                                              opt.max_dataset_size))
        self.fake_paths = sorted(make_dataset(self.dir_fake,
                                              opt.max_dataset_size))
        self.real_size = len(self.real_paths)
        self.fake_size = len(self.fake_paths)
        self.transform = transforms.get_transform(opt, for_val=is_val)
        self.opt = opt

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """
        # read a image given a random integer index
        real_path = self.real_paths[index % self.real_size]  # make sure index is within then range
        if self.opt.no_serial_batches:
            # randomize the index for one domain to avoid fixed pairs
            index_fake = random.randint(0, self.fake_size - 1)
        else:
            # make sure index is within range
            index_fake = index % self.fake_size

        fake_path = self.fake_paths[index_fake]
        real_img = Image.open(real_path).convert('RGB')
        fake_img = Image.open(fake_path).convert('RGB')

        # apply image transformation
        real = self.transform(real_img)
        fake = self.transform(fake_img)

        return {'manipulated': fake,
                'original': real,
                'path_manipulated': fake_path,
                'path_original': real_path
               }

    def __len__(self):
        return max(self.real_size, self.fake_size)
