import tensorflow as tf
import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

parser = argparse.ArgumentParser(description='Save tfrecord format as image format')
parser.add_argument('--tfrecord', required=True, help='Path to tfrecord file')
parser.add_argument('--outdir', required=True, help='Path to save png output')
parser.add_argument('--dataset', required=True, help='which tf record dataset [celebahq|ffhq]')
parser.add_argument('--outsize', type=int, help='resize to this size')
parser.add_argument('--format', type=str, default='png', help='image format')
parser.add_argument('--resize_method', type=str, default='lanczos', help='image format')

args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)
os.makedirs(os.path.join(args.outdir, 'train'), exist_ok=True)
os.makedirs(os.path.join(args.outdir, 'val'), exist_ok=True)
os.makedirs(os.path.join(args.outdir, 'test'), exist_ok=True)

tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
tfr_file = args.tfrecord

resize_methods = {
    'lanczos': Image.LANCZOS,
    'bilinear': Image.BILINEAR
}

if args.dataset == 'celebahq':
    # this will save images in the same order as original celebahq images
    image_list_file = 'resources/celebahq_image_list.txt'
    with open(image_list_file, 'rt') as file:
        lines = [line.split() for line in file]
        fields = dict()
        for idx, field in enumerate(lines[0]):
            type = int if field.endswith('idx') else str
            fields[field] = [type(line[idx]) for line in lines[1:]]

    # determine the train/test/val partitions
    partition_list = 'resources/celeba_list_eval_partition.txt'
    partition_keys={'0':'train', '1':'val', '2':'test'}
    with open(partition_list) as f:
        celeba_partitions={}
        for line in f:
            img, split = line.strip().split()
            img = img.split('.')[0]
            celeba_partitions[img] = partition_keys[split]

    indices = np.array(fields['idx'])
    order = np.arange(len(indices))
    np.random.RandomState(123).shuffle(order)
    for o, record in tqdm(zip(order, tf.python_io.tf_record_iterator(tfr_file, tfr_opt)), total=len(order)):
        ex = tf.train.Example()
        ex.ParseFromString(record)
        shape = ex.features.feature['shape'].int64_list.value
        data = ex.features.feature['data'].bytes_list.value[0]
        im = np.fromstring(data, np.uint8).reshape(shape)
        im = Image.fromarray(np.transpose(im, (1, 2, 0)), 'RGB')
        if args.outsize:
            resizer = resize_methods[args.resize_method]
            im = im.resize((args.outsize, args.outsize), resizer)
        orig_number = fields['orig_file'][o].split('.')[0]
        partition = celeba_partitions[str(orig_number)]
        im.save(os.path.join(args.outdir, partition,
                             '%s.%s' % (orig_number, args.format)))
elif args.dataset == 'ffhq':
    total_count = 70000
    for i, record in tqdm(enumerate(tf.python_io.tf_record_iterator(
        tfr_file, tfr_opt)),total=total_count):
        ex = tf.train.Example()
        ex.ParseFromString(record)
        shape = ex.features.feature['shape'].int64_list.value
        data = ex.features.feature['data'].bytes_list.value[0]
        im = np.fromstring(data, np.uint8).reshape(shape)
        im = Image.fromarray(np.transpose(im, (1, 2, 0)), 'RGB')
        if args.outsize:
            resizer = resize_methods[args.resize_method]
            im = im.resize((args.outsize, args.outsize), resizer)
        if i < 60000:
            partition = 'train'
        elif i < 65000:
            partition = 'val'
        else:
            partition = 'test'
        im.save(os.path.join(args.outdir, partition, '%09d.%s' % (i, args.format)))
else:
    raise NotImplementedError
