from __future__ import print_function
import argparse
import os
import utils.pbar as pbar
import utils.pidfile as pidfile
import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm
import PIL.Image
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='path to pretrained model')
    parser.add_argument('--pretrained', help='downloads pretrained model [celebahq]')
    parser.add_argument('--output_path', required=True, help='path to save generated samples')
    parser.add_argument('--num_samples', type=int, default=100, help='number of samples')
    parser.add_argument('--seed', type=int, default=0, help='random seed for sampling')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for generating samples')
    parser.add_argument("--gpu", default="", type=str, help='GPUs to use (leave blank for CPU only)')
    parser.add_argument("--manipulate", action="store_true", help='add random manipulations to face')
    parser.add_argument("--format", default="jpg", type=str, help='file format to save generated images')
    parser.add_argument("--resize", type=int, help='resizes images to this size before saving')

    opt = parser.parse_args()
    print(opt)
    return opt

def sample(opt):

    # Initialize TensorFlow session.
    tf.InteractiveSession()

    assert(opt.model_path or opt.pretrained), 'specify weights path or pretrained model'

    if opt.model_path:
        raise NotImplementedError
    elif opt.pretrained:
        assert(opt.pretrained  == 'celebahq')
        # make sure to git clone glow repository first
        sys.path.append('resources/glow/demo')
        import model
        eps_std = 0.7
        eps_size = model.eps_size

    rng = np.random.RandomState(opt.seed)
    attr = np.random.RandomState(opt.seed+1)
    tags = []
    amts = []

    for batch_start in tqdm(range(0, opt.num_samples, opt.batch_size)):
        # Generate latent vectors.
        bs = min(opt.num_samples, batch_start + opt.batch_size) - batch_start
        feps = rng.normal(scale=eps_std, size=[bs, eps_size])

        if opt.manipulate:
            tag = attr.randint(len(model._TAGS), size=bs)
            amt = attr.uniform(-1, 1, size=(bs, 1))
            dzs = model.z_manipulate[tag]
            feps = feps + amt * dzs
            tags.append(tag)
            amts.append(amt)

        images = model.decode(feps)

        # Save images as PNG.
        for idx in range(images.shape[0]):
            filename = os.path.join(opt.output_path, 'seed%03d_sample%06d.%s'
                                        % (opt.seed, batch_start + idx,
                                           opt.format))
            im = PIL.Image.fromarray(images[idx], 'RGB')
            if opt.resize:
                im = im.resize((opt.resize, opt.resize), PIL.Image.LANCZOS)
            im.save(filename)

    if opt.manipulate:
        outfile = os.path.join(opt.output_path, 'manipulations.npz')
        np.savez(outfile, tags=np.concatenate(tags), amts=np.concatenate(amts))


if __name__ == '__main__':
    opt = parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    expdir = opt.output_path
    os.makedirs(expdir, exist_ok=True)

    pidfile.exit_if_job_done(expdir)
    sample(opt)
    pidfile.mark_job_done(expdir)
