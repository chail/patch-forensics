"""
modified from PyTorch image folder (https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py)
"""

import torch.utils.data as data

from PIL import Image
import os
import os.path
from utils import util

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif', '.TIF', '.tiff', '.TIFF',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir, max_dataset_size=float("inf")):
    cache = dir.rstrip('/') + '.txt'
    if os.path.isfile(cache):
        print("Using filelist cached at %s" % cache)
        with open(cache) as f:
            images = [line.strip() for line in f]
        # patch up image list with new loading method
        if images[0].startswith(dir):
            print("Using image list from older version")
            image_list = []
            for image in images:
                image_list.append(image)
        else:
            print("Adding prefix to saved image list")
            image_list = []
            prefix = os.path.dirname(dir.rstrip('/'))
            for image in images:
                image_list.append(os.path.join(prefix, image))
        return image_list
    print("Walking directory ...")
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir, followlinks=True)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    image_list = images[:min(max_dataset_size, len(images))]
    with open(cache, 'w') as f:
        prefix = os.path.dirname(dir.rstrip('/')) + '/'
        for i in image_list:
            f.write('%s\n' % util.remove_prefix(i, prefix))
    return image_list

def default_loader(path):
    return Image.open(path).convert('RGB')
