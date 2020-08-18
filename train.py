import time
from options.train_options import TrainOptions
from models import create_model
import numpy as np
from utils.visualizer import Visualizer
import logging
import os
from collections import OrderedDict
from IPython import embed
import torch
import torchvision
import torchvision.transforms as transforms
import pdb
from torch.utils.data import DataLoader
from data.paired_dataset import PairedDataset
from utils import pidfile, util
import utils.logging

def train(opt):
    torch.manual_seed(opt.seed)

    # load the train dataset
    dset = PairedDataset(opt, os.path.join(opt.real_im_path, 'train'),
                         os.path.join(opt.fake_im_path, 'train'))
    # halves batch size since each batch returns both real and fake ims
    dl = DataLoader(dset, batch_size=opt.batch_size // 2,
                    num_workers=opt.nThreads, pin_memory=False,
                    shuffle=True)

    # setup class labeling
    assert(opt.fake_class_id in [0, 1])
    fake_label = opt.fake_class_id
    real_label = 1 - fake_label
    logging.info("real label = %d" % real_label)
    logging.info("fake label = %d" % fake_label)
    dataset_size = 2 * len(dset)
    logging.info('# total images = %d' % dataset_size)
    logging.info('# total batches = %d' % len(dl))

    # setup model and visualizer
    model = create_model(opt)
    epoch, best_val_metric, best_val_ep = model.setup(opt)

    visualizer_losses = model.loss_names + [n + '_val' for n in model.loss_names]
    visualizer = Visualizer(opt, visualizer_losses, model.visual_names)
    total_batches = epoch * len(dl)
    t_data = 0

    now = time.strftime("%c")
    logging.info('================ Training Loss (%s) ================\n' % now)

    while True:
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, ims in enumerate(dl):
            ims_real = ims['original'].to(opt.gpu_ids[0])
            ims_fake = ims['manipulated'].to(opt.gpu_ids[0])
            labels_real = real_label * torch.ones(ims_real.shape[0], dtype=torch.long).to(opt.gpu_ids[0])
            labels_fake = fake_label * torch.ones(ims_fake.shape[0], dtype=torch.long).to(opt.gpu_ids[0])

            batch_im = torch.cat((ims_real, ims_fake), axis=0)
            batch_label = torch.cat((labels_real, labels_fake), axis=0)
            batch_data = dict(ims=batch_im, labels=batch_label)

            iter_start_time = time.time()
            if total_batches % opt.print_freq == 0:
                # time to load data
                t_data = iter_start_time - iter_data_time

            total_batches += 1
            epoch_iter += 1
            model.reset()
            model.set_input(batch_data)
            model.optimize_parameters()

            if epoch_iter % opt.print_freq == 0:
                losses = model.get_current_losses()
                t = time.time() - iter_start_time
                visualizer.print_current_losses(
                    epoch, float(epoch_iter)/len(dl), total_batches,
                    losses, t, t_data)
                visualizer.plot_current_losses(total_batches, losses)

            if epoch_iter % opt.display_freq == 0:
                visualizer.display_current_results(model.get_current_visuals(),
                                                   total_batches)

            if epoch_iter % opt.save_latest_freq == 0:
                logging.info('saving the latest model (epoch %d, total_batches %d)' %
                      (epoch, total_batches))
                model.save_networks('latest', epoch, best_val_metric,
                                    best_val_ep)

            model.reset()
            iter_data_time = time.time()

        # do validation loop at end of each epoch
        model.eval()
        val_start_time = time.time()
        val_losses = validate(model, opt)
        visualizer.plot_current_losses(epoch, val_losses)
        logging.info("Printing validation losses:")
        visualizer.print_current_losses(
            epoch, 0.0, total_batches, val_losses,
            time.time()-val_start_time, 0.0)
        model.train()
        model.reset()
        assert(model.net_D.training)

        # update best model and determine stopping conditions
        if val_losses[model.val_metric + '_val'] > best_val_metric:
            logging.info("Updating best val mode at ep %d" % epoch)
            logging.info("The previous values: ep %d, val %0.2f" %
                         (best_val_ep, best_val_metric))
            best_val_ep = epoch
            best_val_metric = val_losses[model.val_metric + '_val']
            logging.info("The updated values: ep %d, val %0.2f" %
                         (best_val_ep, best_val_metric))
            model.save_networks('bestval', epoch, best_val_metric, best_val_ep)
            with open(os.path.join(model.save_dir, 'bestval_ep.txt'), 'a') as f:
                f.write('ep: %d %s: %f\n' % (epoch, model.val_metric + '_val',
                                           best_val_metric))
        elif epoch > (best_val_ep + 5*opt.patience):
            logging.info("Current epoch %d, last updated val at ep %d" %
                         (epoch, best_val_ep))
            logging.info("Stopping training...")
            break
        elif best_val_metric == 1:
            logging.info("Reached perfect val accuracy metric")
            logging.info("Stopping training...")
            break
        elif opt.max_epochs and epoch > opt.max_epochs:
            logging.info("Reached max epoch count")
            logging.info("Stopping training...")
            break

        logging.info("Best val ep: %d" % best_val_ep)
        logging.info("Best val metric: %0.2f" % best_val_metric)

        # save final plots at end of each epoch
        visualizer.save_final_plots()

        if epoch % opt.save_epoch_freq == 0 and epoch > 0:
            logging.info('saving the model at the end of epoch %d, total batches %d' % (epoch, total_batches))
            model.save_networks('latest', epoch, best_val_metric,
                                best_val_ep)
            model.save_networks(epoch, epoch, best_val_metric, best_val_ep)

        logging.info('End of epoch %d \t Time Taken: %d sec' %
              (epoch, time.time() - epoch_start_time))
        model.update_learning_rate(metric=val_losses[model.val_metric + '_val'])
        epoch += 1

    # save model at the end of training
    visualizer.save_final_plots()
    model.save_networks('latest', epoch, best_val_metric,
                        best_val_ep)
    model.save_networks(epoch, epoch, best_val_metric, best_val_ep)
    logging.info("Finished Training")

def validate(model, opt):
    # --- start evaluation loop --- 
    logging.info('Starting evaluation loop ...')
    model.reset()
    assert(not model.net_D.training)
    val_dset = PairedDataset(opt, os.path.join(opt.real_im_path, 'val'),
                             os.path.join(opt.fake_im_path, 'val'),
                             is_val=True)
    val_dl = DataLoader(val_dset, batch_size=opt.batch_size // 2,
                        num_workers=opt.nThreads, pin_memory=False,
                        shuffle=False)
    val_losses = OrderedDict([(k + '_val', util.AverageMeter())
                              for k in model.loss_names])
    fake_label = opt.fake_class_id
    real_label = 1 - fake_label
    val_start_time = time.time()
    for i, ims in enumerate(val_dl):
        ims_real = ims['original'].to(opt.gpu_ids[0])
        ims_fake = ims['manipulated'].to(opt.gpu_ids[0])
        labels_real = real_label * torch.ones(ims_real.shape[0], dtype=torch.long).to(opt.gpu_ids[0])
        labels_fake = fake_label * torch.ones(ims_fake.shape[0], dtype=torch.long).to(opt.gpu_ids[0])

        inputs = dict(ims=torch.cat((ims_real, ims_fake), axis=0),
                      labels=torch.cat((labels_real, labels_fake), axis=0))

        # forward pass
        model.reset()
        model.set_input(inputs)
        model.test(True)
        losses = model.get_current_losses()

        # update val losses
        for k, v in losses.items():
            val_losses[k + '_val'].update(v, n=len(inputs['labels']))

    # get average val losses
    for k, v in val_losses.items():
        val_losses[k] = v.avg

    return val_losses


if __name__ == '__main__':
    options = TrainOptions(print_opt=False)
    opt = options.parse()

    # lock active experiment directory and write out options
    os.makedirs(os.path.join(opt.checkpoints_dir, opt.name), exist_ok=True)
    pidfile.exit_if_job_done(os.path.join(opt.checkpoints_dir, opt.name))
    options.print_options(opt)

    # configure logging file
    logging_file = os.path.join(opt.checkpoints_dir, opt.name, 'log.txt')
    utils.logging.configure(logging_file, append=False)

    # run train loop
    train(opt)

    # mark done and release lock
    pidfile.mark_job_done(os.path.join(opt.checkpoints_dir, opt.name))
