import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
from utils import rfutil, imutil, pidfile
from tqdm import tqdm
import cv2
from collections import Counter, defaultdict, namedtuple
import matplotlib.pyplot as plt
import argparse
import oyaml as yaml
import sys
sys.path.append('resources/face_parsing_pytorch/')
from model import BiSeNet
import random

# grouped cluster assignments
cluster_assn = {
    0: 'background',
    1: 'skin',
    2: 'brows',
    3: 'brows',
    4: 'eye',
    5: 'eye',
    6: 'eye',
    7: 'ear',
    8: 'ear',
    9: 'ear',
    10: 'nose',
    11: 'mouth',
    12: 'mouth',
    13: 'mouth',
    14: 'neck',
    15: 'neck',
    16: 'clothes',
    17: 'hair',
    18: 'hat',
}

def cluster(args, outpath):

    # network setup
    n_classes = 19
    net = BiSeNet(n_classes=n_classes).cuda()
    save_pth = 'resources/face_parsing_pytorch/res/cp/79999_iter.pth'
    net.load_state_dict(torch.load(save_pth))
    net.eval()
    to_tensor = transforms.Compose([
        transforms.Resize((512, 512), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    subsets = ['reals_easiest', 'fakes_easiest']
    PatchInfo = namedtuple('PatchInfo', ['patch', 'pos', 'file', 'value'])

    for subset in subsets:
        print(subset)
        path = args.path
        cluster_dir = os.path.join(outpath, subset + '_clusters')
        os.makedirs(cluster_dir, exist_ok =True)
        with open(os.path.join(path, subset+'_files.txt')) as f:
            files = [line.strip() for line in f]
        patches = np.load(os.path.join(path, subset + '.npz'))
        which_model_netD = patches['which_model_netD']
        fineSize = patches['finesize']
        rfs = rfutil.find_rf_patches(which_model_netD, fineSize)

        clusters = defaultdict(list)
        clusters_baseline = Counter() # a counter for segmentations of random patches

        # assign each patch to a cluster based on segmentation
        for index, (patch, pos, value, file) in tqdm(enumerate(
            zip(patches['patch'], patches['pos'], patches['value'], files)), total=len(files)):
            image = Image.open(file).convert('RGB')
            with torch.no_grad():
                tensor = to_tensor(image)[None].cuda()
                out = net(tensor)[0]
            parsing = out[0].cpu().numpy().argmax(0)
            interp = cv2.resize(parsing, (fineSize, fineSize), interpolation=cv2.INTER_NEAREST)[None]
            seg_patch = rfutil.get_patch_from_img(interp, rfs[(pos[0], pos[1])], patches['rf'], pad_value=-1)
            # how many pixels of each segmentation class in the patch
            counter_patch = Counter(seg_patch[0].ravel())
            # how many pixels of each segmentation class in the full img
            counter_full = Counter(interp[0].ravel())
            # normalize: omit padding value from normalization
            counter_norm = {k: counter_patch[k] / counter_full[k] for k in counter_patch.keys() if k != -1}
            cluster_id = max(counter_norm, key=counter_norm.get)
            cluster_label = cluster_assn[cluster_id]
            clusters[cluster_label].append(PatchInfo(patch, pos, file, value))

            # pick a random patch for baseline
            random_patch = random.choice(list(rfs.values()))
            seg_patch = rfutil.get_patch_from_img(interp, random_patch, patches['rf'], pad_value=-1)
            counter_patch = Counter(seg_patch[0].ravel())
            counter_full = Counter(interp[0].ravel())
            # omit padding value from normalization
            counter_norm = {k: counter_patch[k] / counter_full[k] for k in counter_patch.keys() if k != -1}
            cluster_id = max(counter_norm, key=counter_norm.get)
            cluster_label = cluster_assn[cluster_id]
            clusters_baseline[cluster_label] += 1

        # plot each cluster in a grid
        counts, labels = [], []
        infos = []
        for index, (k,v) in enumerate(sorted(clusters.items(), key=lambda item: len(item[1]))[::-1]):
            line = '%d: %s, %d patches' % (index, k, len(v))
            print(line)
            infos.append(line)
            counts.append(len(v))
            labels.append(k)
            cluster = np.asarray([patchinfo.patch for patchinfo in v])
            files = [patchinfo.file for patchinfo in clusters[k]]
            normalized = (cluster[-225:] * 0.5) + 0.5 # at most 15x15 grid
            grid = imutil.imgrid(np.uint8(normalized * 255), pad=0, cols=int(np.ceil(np.sqrt(normalized.shape[0]))))
            grid_im = Image.fromarray(grid)
            grid_im.save(os.path.join(cluster_dir, 'cluster_%d.png' % index))
            np.savez(os.path.join(cluster_dir, 'cluster_%d.npz' % index),
                     patch=cluster, rf=patches['rf'],
                     finesize=patches['finesize'],
                     outsize=patches['outsize'],
                     which_model_netD=patches['which_model_netD'],
                     pos=np.array([patchinfo.pos for patchinfo in v]),
                     value=np.array([patchinfo.value for patchinfo in v]))
            with open(os.path.join(cluster_dir, 'cluster_%d.txt' % index), 'w') as f:
                [f.write('%s\n'  % file) for file in files]

        # histogram
        f, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.bar(range(1, len(labels) + 1), counts)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation='vertical')
        ax.set_ylabel('count')
        f.savefig(os.path.join(cluster_dir, 'histogram.pdf'),
                  bbox_inches='tight')

        # write counts to file
        with open(os.path.join(cluster_dir, 'counts.txt'), 'w') as f:
            [f.write('%s\n' % line) for line in infos]

        # write random patch baseline to file
        infos = []
        for index, (k,v) in enumerate(sorted(clusters_baseline.items(), key=lambda item: item[1])[::-1]):
            infos.append('%d: %s, %d patches' % (index, k, v))
        with open(os.path.join(cluster_dir, 'baseline.txt'), 'w') as f:
            [f.write('%s\n' % line) for line in infos]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cluster patches using face segmentation.')
    parser.add_argument('path', type=str, help='path to precomputed top clusters')
    args = parser.parse_args()
    outpath = os.path.join(args.path, 'clusters')
    os.makedirs(outpath, exist_ok =True)
    pidfile.exit_if_job_done(outpath,redo=True)
    cluster(args, outpath)
    pidfile.mark_job_done(outpath)
