import argparse
import oyaml as yaml
import sys
import time
import os
from collections import OrderedDict
from . import util

class Options():
    def __init__(self):
        self.parser = parser = argparse.ArgumentParser()
        self.parser.add_argument('config_file', nargs='?',
                                 type=argparse.FileType(mode='r'))
        self.parser.add_argument('--overwrite_config', action='store_true',
                                 help="overwrite config files if they exist")

    def print_options(self, opt):
        opt_dict = OrderedDict()
        message = ''
        message += '----------------- Options ---------------\n'
        # top level options
        for k, v in sorted(vars(opt).items()):
            if type(v) == argparse.Namespace:
                grouped_k.append((k, v))
                continue
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
            opt_dict[k] = v
        message += '----------------- End -------------------'
        print(message)

        # make experiment directory
        if hasattr(opt, 'checkpoints_dir') and hasattr(opt, 'name'):
            if opt.name != '':
                expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
            else:
                expr_dir = os.path.join(opt.checkpoints_dir)
        else:
            expr_dir ='./'
        os.makedirs(expr_dir, exist_ok=True)

        # save to the disk
        file_name = os.path.join(expr_dir, 'opt.txt')
        if not opt.overwrite_config:
            assert(not os.path.isfile(file_name)), 'config file exists, use --overwrite_config'
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

        file_name = os.path.join(expr_dir, 'opt.yml')
        if not opt.overwrite_config:
            assert(not os.path.isfile(file_name)), 'config file exists, use --overwrite_config'
        with open(file_name, 'wt') as opt_file:
            opt_dict['overwrite_config'] = False # make it false for saving
            yaml.dump(opt_dict, opt_file, default_flow_style=False)

    def parse(self, print_opt=True):

        # parse options
        opt = self.parser.parse_args()

        # get arguments specified in config file
        if opt.config_file:
            data = yaml.load(opt.config_file)
        else:
            data = {}

        # determine which options were specified
        # explicitly with command line args
        option_strings = {}
        for action_group in self.parser._action_groups:
            for action in action_group._group_actions:
                for option in action.option_strings:
                    option_strings[option] = action.dest
        specified_options = set([option_strings[x] for x in
                                 sys.argv if x in option_strings])

        # make namespace
        # by default, take the result from argparse
        # unless was specified in config file and not in command line
        args = {}
        for group in self.parser._action_groups:
            assert(group.title in ['positional arguments',
                                   'optional arguments'])
            group_dict={a.dest: data[a.dest] if a.dest in data
                        and a.dest not in specified_options
                        else getattr(opt, a.dest, None)
                        for a in group._group_actions}
            args.update(group_dict)

        opt = argparse.Namespace(**args)

        delattr(opt, 'config_file')

        # write the configurations to disk
        if print_opt:
            self.print_options(opt)

        self.opt = opt
        return opt
