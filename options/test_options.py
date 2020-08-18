from .base_options import BaseOptions
import oyaml as yaml
import argparse
import sys

class TestOptions(BaseOptions):

    def __init__(self):
        BaseOptions.__init__(self, print_opt=False)
        parser = self.parser
        parser.add_argument('--train_config', type=argparse.FileType(mode='r'), required=True, help='config file saved from model training')
        parser.add_argument('--partition', type=str, default='val', help='val or test')
        parser.add_argument('--dataset_name', type=str, required=True, help="name to describe test dataset when saving results, e.g. celebahq_pgan")
        parser.add_argument('--force_redo', action='store_true', help="force recompute results")

        # for testing model robustness (additional augmentations)
        parser.add_argument('--test_compression', type=int, help='jpeg compression level')
        parser.add_argument('--test_gamma', type=int, help='gamma adjustment level')
        parser.add_argument('--test_blur', type=int, help='blur level')
        parser.add_argument('--test_flip', action='store_true', help='flip all test images')

        # visualizations
        parser.add_argument('--visualize', action='store_true', help='save visualizations when running test')
        parser.add_argument('--average_mode', help='which kind of patch averaging to use for visualizations [vote, before_softmax, after_softmax]')
        parser.add_argument('--topn', type=int, default=100, help='visualize top n')

    def parse(self):
        opt = super().parse()
        train_conf = yaml.load(opt.train_config, Loader=yaml.FullLoader)

        # determine which options were specified
        # explicitly with command line args
        option_strings = {}
        for action_group in self.parser._action_groups:
            for action in action_group._group_actions:
                for option in action.option_strings:
                    option_strings[option] = action.dest
        specified_options = set([option_strings[x] for x in
                                 sys.argv if x in option_strings])

        # make the val options consistent with the train options
        # (e.g. the specified model architecture)
        # but avoid overwriting anything specified in command line
        options_from_train = []
        for k, v in train_conf.items():
            if k in ['real_im_path', 'fake_im_path', 'gpu_ids']:
                # don't overwrite these
                continue
            if getattr(opt, k, None) is None:
                # if the attr is in train but not in base options
                # e.g. learning rate, then skip them
                continue
            if k not in specified_options:
                # overwrite the option if it exists in train
                setattr(opt, k, v)
                options_from_train.append((k, v))

        print("Using the following options from the train configuration file:")
        print(options_from_train)

        # sanity check: make sure partition and data paths is consistent
        # and do some cleaning
        if opt.real_im_path:
            assert(opt.partition in opt.real_im_path)
            opt.real_im_path = opt.real_im_path.rstrip('/')
        if opt.fake_im_path:
            assert(opt.partition in opt.fake_im_path)
            opt.fake_im_path = opt.fake_im_path.rstrip('/')

        opt.load_model = True
        opt.model_seed = 0
        opt.isTrain = False
        return opt


