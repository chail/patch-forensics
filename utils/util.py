import os
import time
from collections import OrderedDict
import torch
import torch.nn as nn
from . import pbar
import numpy as np
from models import networks
from datetime import datetime

def remove_prefix(s, prefix):
    if s.startswith(prefix):
        s = s[len(prefix):]
    return s

def get_subset_dict(in_dict,keys):
    if(len(keys)):
        subset = OrderedDict()
        for key in keys:
            subset[key] = in_dict[key]
    else:
        subset = in_dict
    return subset

def datestring():
    return time.strftime(r"%Y-%m-%d %H:%M:%S")

def format_str_one(v, float_prec=6, int_pad=1):
    if isinstance(v, torch.Tensor) and v.numel() == 1:
        v = v.item()
    if isinstance(v, float):
        return ('{:.' + str(float_prec) + 'f}').format(v)
    if isinstance(v, int) and int_pad:
        return ('{:0' + str(int_pad) + 'd}').format(v)
    return str(v)

def format_str(*args, format_opts={}, **kwargs):
    ss = [format_str_one(arg, **format_opts) for arg in args]
    for k, v in kwargs.items():
        ss.append('{}: {}'.format(k, format_str_one(v, **format_opts)))
    return '\t'.join(ss)

def complete_device(device):
    if not torch.cuda.is_available():
        return torch.device('cpu')
    if type(device) == str:
        device = torch.device(device)
    if device.type == 'cuda' and device.index is None:
        return torch.device(device.type, torch.cuda.current_device())
    return device

def check_timestamp(checkpoint_path, timestamp_path):
    ''' returns True if checkpoint_path timestamp is different
        from timestamp path or timestamp_path doesn't exist'''
    if not os.path.isfile(timestamp_path):
        print("No timestamp found")
        return True
    newtime = os.path.getmtime(checkpoint_path)
    newtime = datetime.fromtimestamp(newtime).strftime('%Y-%m-%d %H:%M:%S')
    with open(timestamp_path) as f:
        oldtime = f.readlines()[0].strip()
    if oldtime != newtime:
        print("Timestamp out of date")
        return True
    print("Timestamp is correct")
    return False

def update_timestamp(checkpoint_path, timestamp_path):
    ''' write the last modified date of checkpoint_path to the
        the file timestamp_path '''
    newtime = os.path.getmtime(checkpoint_path)
    newtime = datetime.fromtimestamp(newtime).strftime('%Y-%m-%d %H:%M:%S')
    with open(timestamp_path, 'w') as f:
        f.write('%s' % newtime)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
