from options.test_options import TestOptions
from models import create_model
import numpy as np
import os
import torch
from utils import pidfile
from utils import rfutil
from utils import imutil
from utils import util
from data.unpaired_dataset import UnpairedDataset
from torch.utils.data import DataLoader
import heapq
from collections import namedtuple, OrderedDict, Counter
from itertools import count
import matplotlib.pyplot as plt
from PIL import Image


torch.backends.cudnn.benchmark = True

def run_patch_topn(opt, output_dir):
    assert opt.model == 'patch_discriminator' # only works on patch models
    model = create_model(opt)
    model.setup(opt)
    model.eval()

    # find patch indices
    rf = rfutil.find_rf_model(opt.which_model_netD)
    print("Receptive field %d" % rf)
    patches = rfutil.find_rf_patches(opt.which_model_netD, opt.fineSize)

    fake_label = opt.fake_class_id
    real_label = 1 - fake_label

    # operations
    softmax = torch.nn.Softmax(dim=1)
    PatchInfo = namedtuple('PatchInfo', ['patch', 'pos', 'file', 'value'])

    for data_path, label, name in zip([opt.real_im_path, opt.fake_im_path],
                                      [real_label, fake_label],
                                      ['reals', 'fakes']):
        dset = UnpairedDataset(opt, data_path, is_val=True)
        dl = DataLoader(dset, batch_size=opt.batch_size, shuffle=False,
                        num_workers=opt.nThreads, pin_memory=False)
        transform = dset.transform
        heap_easiest = []

        # heappush will error if there are ties in value
        # use counter to break ties in the heap
        tiebreaker = count()

        for i, data in enumerate(dl):
            # set model inputs
            ims = data['img'].to(opt.gpu_ids[0])
            assert(ims.shape[-1] == opt.fineSize)
            pred_labels = label * torch.ones(ims.shape[0], dtype=torch.long).cuda()
            inputs = dict(ims=ims, labels=pred_labels)

            # forward pass
            model.reset()
            model.set_input(inputs)
            model.test(True)

            # get model outputs
            with torch.no_grad():
                model_out = softmax(model.pred_logit).detach().cpu().numpy()
                assert(np.ndim(model_out) == 4) # for patch model

            for pred, path, img in zip(model_out, data['path'], ims):
                img = img.cpu().numpy()
                pred = pred[label, :, :] # get class prediction
                patch_values = np.sort(pred, axis=None)
                random_tiebreak = np.random.random(pred.size)
                # if values are the same, take a random patch among
                # everything that has the same values
                # lexsort does second entry then first entry for sort order
                tiebreak_argsort = np.lexsort((random_tiebreak.ravel(),
                                               pred.ravel()))
                ylocs, xlocs = np.unravel_index(tiebreak_argsort,
                                                pred.shape)
                num = 1 if opt.unique else opt.topn
                # just iterate through top predictions for efficiency
                for value, yloc, xloc in zip(patch_values[-num:],
                                             ylocs[-num:],
                                             xlocs[-num:]):
                    assert(pred[yloc, xloc] == value)
                    if len(heap_easiest) < opt.topn or value > heap_easiest[0][0]:
                        patch_pos = (yloc, xloc)
                        patch_file = path
                        slices = patches[(yloc, xloc)]
                        patch_img = rfutil.get_patch_from_img(img, slices, rf)
                        patch_info = PatchInfo(patch_img, patch_pos,
                                               patch_file, value)
                        if len(heap_easiest) < opt.topn:
                             heapq.heappush(heap_easiest, (value, next(tiebreaker), patch_info))
                        else:
                            heapq.heappushpop(heap_easiest, (value, next(tiebreaker), patch_info))

        # aggregate and save results (easiest)
        heap_easiest_sorted = sorted(heap_easiest)
        infos = OrderedDict(
            patch=np.array([h[2].patch for h in heap_easiest_sorted]),
            pos=np.array([h[2].pos for h in heap_easiest_sorted]),
            value=np.array([h[2].value for h in heap_easiest_sorted]),
            outsize=pred.shape, rf=rf, finesize=opt.fineSize,
            which_model_netD=opt.which_model_netD)
        np.savez(os.path.join(output_dir, name + '_easiest.npz'), **infos)

        with open(os.path.join(output_dir, name+'_easiest_files.txt'), 'w') as f:
            [f.write('%s\n' % h[2].file) for h in heap_easiest_sorted]

        # grid image of the easiest patches
        normalized = (infos['patch'] * 0.5) + 0.5
        grid = imutil.imgrid(np.uint8(normalized * 255), pad=0, cols=
                             int(np.ceil(np.sqrt(normalized.shape[0]))))
        im = Image.fromarray(grid)
        im.save(os.path.join(output_dir, name + '_easiest_grid.jpg'))


if __name__ == '__main__':
    options = TestOptions()
    # additional options for top n patches
    options.parser.add_argument('--unique', action='store_true', help='take only 1 patch per image when computing top n')
    opt = options.parse()
    print("Calculating patches from model: %s epoch %s" % (opt.name, opt.which_epoch))
    print("On dataset (real): %s" % (opt.real_im_path))
    print("And dataset (fake): %s" % (opt.fake_im_path))
    expdir = opt.name
    dataset_name = opt.dataset_name
    output_dir = os.path.join(opt.results_dir, expdir, opt.partition,
                              'epoch_%s' % opt.which_epoch, dataset_name,
                              'patches_top%d' % opt.topn)
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # check if checkpoint is out of date
    redo = opt.force_redo
    ckpt_path = os.path.join(opt.checkpoints_dir, opt.name, '%s_net_D.pth' % opt.which_epoch)
    timestamp_path = os.path.join(output_dir, 'timestamp_%s_net_D.txt' % opt.which_epoch)
    if util.check_timestamp(ckpt_path, timestamp_path):
        redo = True
        util.update_timestamp(ckpt_path, timestamp_path)
    pidfile.exit_if_job_done(output_dir, redo=True) # redo=redo)
    run_patch_topn(opt, output_dir)
    pidfile.mark_job_done(output_dir)

