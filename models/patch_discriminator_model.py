import torch
import torch.nn as nn
from collections import OrderedDict
from utils import renormalize, imutil
from .base_model import BaseModel
from .networks import networks
import numpy as np
import logging
import cv2
from PIL import Image
from collections import namedtuple

class PatchDiscriminatorModel(BaseModel):

    def name(self):
        return 'PatchDiscriminatorModel'

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out. 
        self.loss_names = ['loss_D']
        self.loss_names += ['acc_D_raw', 'acc_D_voted', 'acc_D_avg']
        self.val_metric = 'acc_D_raw'

        # specify the images you want to save/display. 
        self.visual_names = ['fake_0', 'fake_1', 'fake_2', 'fake_3', 'fake_4',
                             'real_0', 'real_1', 'real_2', 'real_3', 'real_4',
                             'vfake_0', 'vfake_1', 'vfake_2', 'vfake_3', 'vfake_4',
                             'vreal_0', 'vreal_1', 'vreal_2', 'vreal_3', 'vreal_4']

        # specify the models you want to save to the disk. 
        self.model_names = ['D']

        # load/define networks
        torch.manual_seed(opt.seed) # set model seed
        self.net_D = networks.define_patch_D(opt.which_model_netD,
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
        # logit shape should be N2HW
        assert(len(self.pred_logit.shape) == 4)
        assert(self.pred_logit.shape[1] == 2)
        n, c, h, w = self.pred_logit.shape
        labels = self.labels.view(-1, 1, 1).expand(n, h, w)
        predictions = self.pred_logit
        self.loss_D = self.criterionCE(predictions, labels)
        self.acc_D_raw = torch.mean(torch.eq(labels, torch.argmax(
            predictions, dim=1)).float())
        # voted acc is forcing each patch into a 0/1 decision,
        # and taking the average
        votes = torch.mode(torch.argmax(predictions, dim=1).view(n, -1))[0]
        self.acc_D_voted = torch.mean(torch.eq(self.labels, votes).float())
        # average acc is averaging each softmaxed patch prediction, and 
        # taking the argmax
        avg_preds = torch.argmax(self.softmax(self.pred_logit)
                                 .mean(dim=(2,3)), dim=1)
        self.acc_D_avg = torch.mean(torch.eq(self.labels,
                                             avg_preds).float())

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
        with torch.no_grad():
            fake_ims_overlay = self.softmax(self.pred_logit[
                self.labels == self.opt.fake_class_id])
        real_ims = self.ims[self.labels != self.opt.fake_class_id]
        with torch.no_grad():
            real_ims_overlay = self.softmax(self.pred_logit[
                self.labels != self.opt.fake_class_id])
        for i in range(min(5, len(fake_ims))):
            im = renormalize.as_tensor(
                fake_ims[[i], :, :, :], source='zc', target='pt')
            visual_ret['fake_%d' % i] = im
            visual_ret['vfake_%d' % i] = self.overlay_visual(
                im.detach().cpu().numpy().squeeze(),
                fake_ims_overlay[i, 1, :, :].detach().cpu().numpy(),
                to_tensor=True
            )
        for i in range(min(5, len(real_ims))):
            im = renormalize.as_tensor(
                real_ims[[i], :, :, :], source='zc', target='pt')
            visual_ret['real_%d' % i] = im
            visual_ret['vreal_%d' % i] = self.overlay_visual(
                im.detach().cpu().numpy().squeeze(),
                real_ims_overlay[i, 1, :, :].detach().cpu().numpy(),
                to_tensor=True
            )
        return visual_ret

    def reset(self):
        # for debugging .. clear all the cached variables
        self.loss_D = None
        self.acc_D = None
        self.acc_D_raw = None
        self.acc_D_voted = None
        self.acc_D_avg = None
        self.ims = None
        self.labels = None
        self.pred_logit = None

    def overlay_visual(self, im_np, overlay, to_tensor=False):
        # im : np array, (3, h, w)
        # overlay: np array (h', w')
        (h, w) = im_np.shape[1:]
        overlay = np.uint8(255 * overlay)
        overlay = cv2.resize(overlay, (w, h))
        heatmap = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
        heatmap = heatmap/255 # range [0, 1]
        # change heatmap to RGB, and channel to CHW
        heatmap = heatmap[:,:,::-1].transpose(2, 0, 1)
        result = heatmap * 0.3 + im_np * 0.5
        if to_tensor:
            return torch.from_numpy(result).float().to(self.device)[None]
        else:
            im_out = np.uint8(result*255)
            return im_out

    def get_predictions(self):
        Predictions = namedtuple('predictions', ['vote', 'before_softmax',
                                                 'after_softmax', 'raw'])
        with torch.no_grad():
            n = self.pred_logit.shape[0]
            # vote_predictions probability is a tally of the patch votes
            votes = torch.argmax(self.pred_logit, dim=1).view(n, -1)
            vote_predictions = torch.mean(votes.float(), axis=1)
            vote_predictions = torch.stack([1-vote_predictions,
                                            vote_predictions], axis=1)
            before_softmax_predictions = self.softmax(
                torch.mean(self.pred_logit, dim=(-1, -2)))
            after_softmax_predictions = torch.mean(
                self.softmax(self.pred_logit), dim=(-1, -2))
            patch_predictions = self.softmax(self.pred_logit)
        return Predictions(vote_predictions.cpu().numpy(),
                           before_softmax_predictions.cpu().numpy(),
                           after_softmax_predictions.cpu().numpy(),
                           patch_predictions.cpu().numpy())

    def visualize(self, pred_outputs, pred_paths, labels, transform,
                  target_label, dirname, n=100):
        import shutil
        import os
        # take a list of the predictions, corresponding image path
        # and label for each prediction label, and a target label
        # to visualize --> draws heatmaps for the top hardest/easiest
        # n images
        os.makedirs(dirname, exist_ok=True)
        idx = np.where(labels == target_label)[0]
        pred_outputs = pred_outputs[idx]
        pred_paths = [pred_paths[i] for i in idx]
        order = np.argsort(pred_outputs[:, target_label])
        softmax = self.softmax
        for indices, desc in zip([order[:n], order[-n:][::-1]], ['hardest', 'easiest']):
            os.makedirs(os.path.join(dirname, desc), exist_ok=True)

            # average heatmap aggregator
            avg_heatmap = []

            # write to file
            with open(os.path.join(dirname, desc, 'preds.txt'), 'w') as f:
                for i in indices:
                    output = ','.join(['%0.6f' % s for s in pred_outputs[i]])
                    f.write('{},{},{}\n'.format(i, output, pred_paths[i]))

            # save visualizations
            for c, i in enumerate(indices, 1):
                im_orig = Image.open(pred_paths[i])
                im = transform(im_orig).to(self.device)[None] # make it tensor
                im_orig = renormalize.as_image(im[0])
                im_orig.save(os.path.join(dirname, desc, '%03d_orig.png' % c))
                w, h = im_orig.size

                with torch.no_grad():
                    # get heatmap for visualization, NCHW
                    pred_out = softmax(self.net_D(im)).cpu().numpy()
                # green border if correct prediction, otherwise red
                predicted_class = np.argmax(pred_outputs[i])
                prob_real = pred_outputs[i][1 - self.opt.fake_class_id]
                color = (0, 255, 0) if predicted_class == target_label else (255, 0, 0)

                # patch probability visualization
                for label in [1]: # just plot P(real)
                    heatmap = pred_out[0, label, :, :]
                    np.savez(os.path.join(dirname, desc, '%03d_heatmap_%d.npz'
                                          % (c, label)), heatmap=heatmap)
                    out = pred_out[0, label, :, :]
                    avg_heatmap.append(out)
                    # normalizes between min and max
                    out = out - np.min(out)
                    out = out / (np.max(out) + 1e-6) # bit of tolerance in div 
                    out_img = np.uint8(255 * out)
                    out_img= cv2.resize(out_img, (w, h))
                    heatmap = cv2.applyColorMap(out_img, cv2.COLORMAP_JET)
                    heatmap = heatmap[:,:,::-1] # change heatmap to RGB
                    result = heatmap * 0.3 + np.asarray(im_orig) * 0.5

                    # save output
                    cv2.rectangle(result, (0, 0), (result.shape[1], result.shape[0]),
                                  color=color, thickness=2)
                    cv2.putText(result, "p(real)=%0.2f" % prob_real, (0, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255))
                    result = Image.fromarray(np.uint8(result))
                    result.save(os.path.join(dirname, desc, '%03d_patch_%d.png' % (c, label)))

            # webpage
            lbname = os.path.join(dirname, desc, '+lightbox.html')
            if not os.path.exists(lbname):
                shutil.copy('utils/lightbox.html', lbname)

            # save average heatmap
            avg_heatmap = np.mean(np.stack(avg_heatmap), axis=0)
            np.savez(os.path.join(dirname, desc, 'heatmap_avg.npz'),
                     heatmap=avg_heatmap)
            avg_heatmap = avg_heatmap - np.min(avg_heatmap)
            avg_heatmap = avg_heatmap / np.max(avg_heatmap)
            avg_heatmap = np.uint8(255*avg_heatmap)
            # note that this resizes it to original image size
            avg_heatmap = cv2.resize(avg_heatmap, (w, h))
            avg_heatmap = cv2.applyColorMap(avg_heatmap, cv2.COLORMAP_JET)
            avg_heatmap = avg_heatmap[:,:,::-1] # change heatmap to RGB
            Image.fromarray(avg_heatmap).save(
                os.path.join(dirname, desc, 'heatmap_avg.png'))
