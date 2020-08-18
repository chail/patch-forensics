import torch
import torch.nn as nn
from collections import OrderedDict
from utils import renormalize, imutil
from .base_model import BaseModel
from .networks import networks
import numpy as np
import logging
from collections import namedtuple

class BasicDiscriminatorModel(BaseModel):

    def name(self):
        return 'BasicDiscriminatorModel'

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. 
        self.loss_names = ['loss_D']
        self.loss_names += ['acc_D']
        self.val_metric = 'acc_D'

        # specify the images you want to save/display. 
        self.visual_names = ['fake_0', 'fake_1', 'fake_2', 'fake_3', 'fake_4',
                             'real_0', 'real_1', 'real_2', 'real_3', 'real_4']

        # specify the models you want to save to the disk. 
        self.model_names = ['D']

        # load/define networks
        torch.manual_seed(opt.seed) # set model seed
        self.net_D = networks.define_D(opt.which_model_netD,
                                       opt.init_type, self.gpu_ids)
        self.criterionCE = nn.CrossEntropyLoss().to(self.device)
        self.softmax = torch.nn.Softmax(dim=1)

        if self.isTrain:
            self.optimizers['D'] = torch.optim.Adam(
                self.net_D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def set_input(self, input):
        self.ims = input['ims'].to(self.device)
        self.labels = input['labels'].to(self.device)

    def forward(self):
        self.pred_logit = self.net_D(self.ims)

    def compute_losses_D(self):
        self.loss_D = self.criterionCE(self.pred_logit, self.labels)
        self.acc_D = torch.mean(torch.eq(self.labels, torch.argmax(
            self.pred_logit, dim=1)).float())

    def backward_D(self):
        self.compute_losses_D()
        self.loss_D.backward()

    def optimize_parameters(self):
        self.optimizers['D'].zero_grad()
        self.forward()
        self.backward_D()
        self.optimizers['D'].step()

    def get_current_visuals(self):
        from collections import OrderedDict
        visual_ret = OrderedDict()
        fake_ims = self.ims[self.labels == self.opt.fake_class_id]
        real_ims = self.ims[self.labels != self.opt.fake_class_id]
        for i in range(min(5, len(fake_ims))):
            visual_ret['fake_%d' % i] = renormalize.as_tensor(
                fake_ims[[i], :, :, :], source='zc', target='pt')
        for i in range(min(5, len(real_ims))):
            visual_ret['real_%d' % i] = renormalize.as_tensor(
                real_ims[[i], :, :, :], source='zc', target='pt')
        return visual_ret

    def reset(self):
        # for debugging .. clear all the cached variables
        self.loss_D = None
        self.acc_D = None
        self.ims = None
        self.labels = None
        self.pred_logit = None

    def get_predictions(self, *args):
        # makes it consistent with patch discriminator outputs
        Predictions = namedtuple('predictions', ['vote', 'before_softmax',
                                                 'after_softmax', 'raw'])
        with torch.no_grad():
            predictions = self.softmax(self.pred_logit).cpu().numpy()
        return Predictions(None, None, None, predictions)

    def visualize(self, pred_outputs, pred_paths, labels, transform,
                  target_label, dirname, n=100, **kwargs):
        print("Visualization only implemented for patch-based models")
        raise NotImplementedError
