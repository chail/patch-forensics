import logging
import os
import numpy as np
import time
from tensorboardX import SummaryWriter
import torchvision.utils as vutils
import torch

class LinePlotter(object):
    def __init__(self, writer, tag):
        self.writer = writer
        self.tag = tag

    def plot(self, x, data, walltime=None):
        # x is a scalar
        # data is a scalar
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': []}
        self.plot_data['X'].append(x)
        self.plot_data['Y'].append(data)
        self.writer.add_scalar(self.tag, data, x, walltime)

    def save_final_plot(self, save_dir):
        # save_dir = writer.logdir
        save_path = os.path.join(save_dir, '{}'.format(self.tag.replace('/', '_')))
        os.makedirs(save_path, exist_ok=True)
        if hasattr(self, 'plot_data'):
            save_data = dict(X=np.array(self.plot_data['X']),
                             Y=np.array(self.plot_data['Y'])
                            )
            np.savez(save_path + '.npz', **save_data)
        logging.info('Saved to {}'.format(save_path))

class ImageGridPlotter(object):
    def __init__(self, writer, ncols, grid=False):
        self.ncols = ncols
        self.writer = writer
        self.grid = grid

    def plot(self, visuals, niter=0):
        ncols = self.ncols
        ncols = min(ncols, len(visuals))
        if self.grid:
            images = []
            labels = '|'
            idx = 0
            for label, im in visuals.items():
                images.append(im[0])
                labels += label + '|'
                idx += 1
                if idx % ncols == 0 and idx > 0:
                    labels += '||'
            blank_image = torch.ones_like(images[0])
            while idx % ncols != 0:
                images.append(blank_image)
                idx += 1
                labels += ' |'
            self.writer.add_text('Visuals Labels', labels, niter)
            x = vutils.make_grid(images, normalize=True, nrow=ncols) # scale_each=True)
            self.writer.add_image('Visuals', x, niter)
        else:
            for label, im in visuals.items():
                x = vutils.make_grid([im[0]], normalize=True)
                self.writer.add_image(label, x, niter)
