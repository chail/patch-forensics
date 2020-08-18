import os.path
import torch.utils.data as data
from .dataset_util import make_dataset
from PIL import Image
import numpy as np
import torch
from . import transforms

class UnpairedDataset(data.Dataset):
    """A dataset class for loading images within a single folder
    """
    def __init__(self, opt, im_path, is_val=False):
        """Initialize this dataset class.

        Parameters:
            opt -- experiment options
            im_path -- path to folder of images
            is_val -- is this training or validation? used to determine
            transform
        """
        super().__init__()
        self.dir = im_path
        self.paths = sorted(make_dataset(self.dir, opt.max_dataset_size))
        self.size = len(self.paths)
        assert(self.size > 0)
        self.transform = transforms.get_transform(opt, for_val=is_val)
        self.opt = opt

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing
        """
        # read a image given a random integer index
        path = self.paths[index]
        img = Image.open(path).convert('RGB')

        # apply image transformation
        img = self.transform(img)

        return {'img': img,
                'path': path
               }

    def __len__(self):
        return self.size
