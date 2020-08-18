import numpy as np
import os
import logging
from tensorboardX import SummaryWriter

class Visualizer():
    def __init__(self, opt, loss_names, visual_names=None):
        from . import tensorboard_utils as tb_utils
        self.name = opt.name
        self.opt = opt
        self.visual_names = visual_names
        # check that tensorboard history does not exist
        tb_path = os.path.join('runs', self.name)
        if os.path.isdir(tb_path):
            logging.info('Found existing tensorboard history at %s' % tb_path)
            if not opt.overwrite_config:
                logging.info('Use --overwrite_config to write to existing tensorboard history')
                exit(0)
        self.writer = SummaryWriter(logdir=tb_path)
        self.plotters = []
        for name in loss_names:
            setattr(self, name + '_plotter', tb_utils.LinePlotter(
                self.writer, name.replace('_', '/', 1)))
            self.plotters.append(getattr(self, name + '_plotter'))
        self.imgrid = tb_utils.ImageGridPlotter(
            self.writer, ncols=5, grid=True)

    # |visuals|: dictionary of image tensors to display 
    def display_current_results(self, visuals, epoch):
        # show images in the browser
        self.imgrid.plot(visuals, epoch)

    # losses: dictionary of error labels and values
    def plot_current_losses(self, niter, losses):
        for k, v in losses.items():
            plotter = getattr(self, k + '_plotter')
            plotter.plot(niter, v)

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, total_steps, losses, t, t_data, prefix=''):
        message = '(epoch: %d, iters: %.3f, time: %.3f, data: %.3f) ' % (epoch, iters, t, t_data)
        message += prefix
        message += ' '
        for k, v in losses.items():
            message += '%s: %.3f, ' % (k, v)

        logging.info('%s' % message)
        logging.info('Total batches: %0.2f k\n' % (total_steps / 1000))

    def save_final_plots(self):
        save_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, 'visualize')
        for plotter in self.plotters:
            plotter.save_final_plot(save_dir)
