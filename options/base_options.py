import argparse
import os
from utils import util
from utils import options
import torch
import models

class BaseOptions(options.Options):
    def __init__(self, print_opt=True):
        options.Options.__init__(self)
        self.isTrain = False # train_options will change this
        self.print_opt = print_opt
        parser = self.parser

        # model setup
        parser.add_argument('--model', type=str, default='basic_discriminator', help='chooses which model to use')
        parser.add_argument('--which_model_netD', type=str, default='resnet18', help='selects model to use for netD')
        parser.add_argument('--fake_class_id', type=int, default=0, help='class id of fake ims')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_model', action='store_true', help='load the latest model')
        parser.add_argument('--seed', type=int, default=0, help='torch.manual_seed value')
        parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')

        # image loading
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        parser.add_argument('--nThreads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
        parser.add_argument('--real_im_path', type=str, help='path to real images')
        parser.add_argument('--fake_im_path', type=str, help='path to fake images')
        parser.add_argument('--no_serial_batches', action='store_true', help='if not specified, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help="Maximum number of samples to use in dataset")

        # checkpoint saving and naming 
        parser.add_argument('--name', type=str, default='', help='name of the experiment. it decides where to store samples and models')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
        parser.add_argument('--prefix', default='', type=str, help='customized prefix: opt.name = prefix + opt.name: e.g., {model}_{which_model_netG}_size{loadSize}')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')

    def parse(self):

        opt = options.Options.parse(self, print_opt=False)

        # modify model-related parser options 
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        self.parser = model_option_setter(self.parser)
        opt = options.Options.parse(self, print_opt=False)

        opt.isTrain = self.isTrain

        # default model name
        if opt.name == '':
            opt.name = '{model}_{which_model_netD}_size{fineSize}'.format(**vars(opt))
        else:
            opt.name = opt.name.format(**vars(opt))

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix
            opt.suffix = ''

        # process opt.prefix
        if opt.prefix:
            prefix = (opt.prefix.format(**vars(opt))) if opt.prefix != '' else ''
            prefix += '-'
            opt.name = prefix + opt.name
            opt.prefix = ''

        # print options after name/prefix/suffix is modified
        if self.print_opt:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids
        if isinstance(opt.gpu_ids, str):
            str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
            if len(opt.gpu_ids) > 0 and torch.cuda.is_available():
                torch.cuda.set_device(opt.gpu_ids[0])

        # check both image paths are specified
        assert(opt.real_im_path and opt.fake_im_path)
        return opt
