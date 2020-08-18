import os
import torch
from collections import OrderedDict
from .networks import netutils
from .networks import networks
import logging


class BaseModel():
    @staticmethod
    def modify_commandline_options(parser):
        networks.modify_commandline_options(parser)
        return parser

    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        torch.backends.cudnn.benchmark = True
        self.loss_names = []
        self.model_names = []
        self.visual_names = []
        self.image_paths = []
        self.optimizers = {}

    def name(self):
        return 'BaseModel'

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # load and print networks; create schedulers
    def setup(self, opt, parser=None):
        current_ep = 0
        best_val_metric, best_val_ep = 0, 0
        self.print_networks()
        if self.isTrain:
            self.schedulers = {k: netutils.get_scheduler(optim, opt) for
                               (k, optim) in self.optimizers.items()}
        if not self.isTrain or opt.load_model:
            current_ep, best_val_metric, best_val_ep  = self.load_networks(opt.which_epoch)
            if opt.which_epoch not in ['latest', 'bestval']:
                # checkpoint was saved at end of epoch
                current_ep += 1
        return current_ep, best_val_metric, best_val_ep

    # make models eval mode 
    def eval(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.eval()

    # make models train mode
    def train(self):
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                net.train()

    # used in test time, wrapping `forward` in no_grad() so we don't save
    # intermediate steps for backprop
    def test(self, compute_losses=False):
        with torch.no_grad():
            self.forward()
            if(compute_losses):
                self.compute_losses_D()

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def optimize_parameters(self):
        pass

    # update learning rate (called once every epoch)
    def update_learning_rate(self, metric=None):
        for k, scheduler in self.schedulers.items():
            if metric is not None:
                assert self.opt.lr_policy in ['plateau', 'constant']
                scheduler.step(metric)
            else:
                scheduler.step()
        for k, optim in self.optimizers.items():
            logging.info('learning rate net_%s = %0.7f' % (k, optim.param_groups[0]['lr']))

    # return visualization images. train.py will display these images
    def get_current_visuals(self):
        visual_ret = OrderedDict()
        for name in self.visual_names:
            assert(isinstance(name, str))
            visual_ret[name] = getattr(self, name)
        return visual_ret

    # return training losses/errors. train.py will print out these errors as debugging information
    def get_current_losses(self):
        errors_ret = OrderedDict()
        for name in self.loss_names:
            assert(isinstance(name, str))
            # float(...) works for both scalar tensor and float number
            errors_ret[name] = float(getattr(self, name))
        return errors_ret

    # save models to the disk
    def save_networks(self, save_name, current_ep,
                      best_val_metric, best_val_ep):
        for name in self.model_names:
            assert(isinstance(name, str))
            save_filename = '%s_net_%s.pth' % (save_name, name)
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, 'net_' + name)
            if isinstance(net, torch.nn.DataParallel):
                sd = net.module.state_dict()
            else:
                sd = net.state_dict()

            optim = self.optimizers[name].state_dict()
            sched = self.schedulers[name].state_dict()

            checkpoint = dict(state_dict=sd, optimizer=optim,
                              scheduler=sched, epoch=current_ep,
                              best_val_metric=best_val_metric,
                              best_val_ep=best_val_ep)
            torch.save(checkpoint, save_path)

    # load models from the disk
    def load_networks(self, save_name):
        for name in self.model_names:
            assert(isinstance(name, str))
            load_filename = '%s_net_%s.pth' % (save_name, name)
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, 'net_' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            checkpoint = torch.load(load_path, map_location=str(self.device))
            state_dict = checkpoint['state_dict']
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata
            net.load_state_dict(state_dict)

            if self.isTrain:
                print('restoring optimizer and scheduler for %s' % name)
                self.optimizers[name].load_state_dict(checkpoint['optimizer'])
                self.schedulers[name].load_state_dict(checkpoint['scheduler'])
            current_ep = checkpoint['epoch']
            best_val_metric = checkpoint['best_val_metric']
            best_val_ep = checkpoint['best_val_ep']
        return current_ep, best_val_metric, best_val_ep

    # print network information
    def print_networks(self):
        print('---------- Networks initialized -------------')
        for name in self.model_names:
            if isinstance(name, str):
                net = getattr(self, 'net_' + name)
                num_params = 0
                for param in net.parameters():
                    num_params += param.numel()
                print(net)
                print('[Network %s] Total number of parameters : %.3f M' % (name, num_params / 1e6))
        print('-----------------------------------------------')

    # set requires_grad=False to avoid computation
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
