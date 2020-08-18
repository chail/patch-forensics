from __future__ import print_function
import argparse
import os
import utils.pbar as pbar
import utils.pidfile as pidfile
import numpy as np
import dill as pickle
import tensorflow as tf
import PIL.Image
from tqdm import tqdm
import sys
import albumentations as A
sys.path.append('resources/progressive_growing_of_gans')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='path to pretrained model')
    parser.add_argument('--output_path', required=True, help='path to save generated samples')
    parser.add_argument('--num_samples', type=int, default=100, help='number of samples')
    parser.add_argument('--seed', type=int, default=0, help='random seed for sampling')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for generating samples')
    parser.add_argument("--gpu", default="", type=str, help='GPUs to use (leave blank for CPU only)')
    parser.add_argument("--format", default="jpg", type=str, help='file format to save generated images')
    parser.add_argument("--resize", type=int, help='resizes images to this size before saving')
    parser.add_argument("--quality", type=int, help='compression quality')

    opt = parser.parse_args()
    print(opt)
    return opt

def sample(opt):

    # Initialize TensorFlow session.
    tf.InteractiveSession()

    # Import official CelebA-HQ networks.
    with open(opt.model_path, 'rb') as file:
            G, D, Gs = pickle.load(file)

    rng = np.random.RandomState(opt.seed)

    for batch_start in tqdm(range(0, opt.num_samples, opt.batch_size)):
        # Generate latent vectors.
        bs = min(opt.num_samples, batch_start + opt.batch_size) - batch_start
        latents = rng.randn(bs, *Gs.input_shapes[0][1:])
        # Generate dummy labels (not used by the official networks).
        labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])

        # Run the generator to produce a set of images.
        images = Gs.run(latents, labels)

        # Convert images to PIL-compatible format.
        images = np.clip(np.rint((images + 1.0) / 2.0 * 255.0), 0.0,
                         255.0).astype(np.uint8) # [-1,1] => [0,255]
        images = images.transpose(0, 2, 3, 1) # NCHW => NHWC

        # Save images as PNG.
        for idx in range(images.shape[0]):
            filename = os.path.join(opt.output_path, 'seed%03d_sample%06d.%s'
                                        % (opt.seed, batch_start + idx,
                                           opt.format))
            im = PIL.Image.fromarray(images[idx], 'RGB')
            if opt.resize:
                im = im.resize((opt.resize, opt.resize), PIL.Image.LANCZOS)
            if opt.quality:
                aug = A.augmentations.transforms.JpegCompression(p=1)
                w, h = im.size
                im_np = np.asarray(im.resize((1024, 1024), PIL.Image.LANCZOS))
                im = PIL.Image.fromarray(aug.apply(im_np, quality=opt.quality))
                im = im.resize((w, h), PIL.Image.LANCZOS)
            im.save(filename)


if __name__ == '__main__':
    opt = parse_args()
    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu
    expdir = opt.output_path
    os.makedirs(expdir, exist_ok=True)

    pidfile.exit_if_job_done(expdir)
    sample(opt)
    pidfile.mark_job_done(expdir)
