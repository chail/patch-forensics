import torch
import torch.nn as nn
from models.networks import networks
import numpy as np
import os
from collections import OrderedDict
from tqdm import tqdm

def rf(output_size, ksize, stride):
    return (output_size -1) * stride + ksize

def find_downsamples(model):
    downsamples = []
    def find_downsamples_rec(model, layers):
        downsamples = (nn.Conv2d, nn.MaxPool2d, nn.AvgPool2d)
        for block in model.children():
            if isinstance(block, downsamples):
                layers.append(block)
            layers = find_downsamples_rec(block, layers)
        return layers
    find_downsamples_rec(model, downsamples)
    return downsamples

def find_downsamples_resnet(model):
    downsamples = ['conv', 'maxpool']
    # skips the convs in the downsample block 
    # which downsamples the input to add to residual features
    # it has smaller rf
    return [(n, l) for n, l in model.named_modules() if
            any([d in n for d in downsamples])]

def find_downsamples_xceptionnet(model):
    downsamples = ['conv', 'maxpool']
    ds_list = []
    for n, l in model.named_modules():
        if any([n.startswith(d) for d in downsamples]):
            ds_list.append((n, l))
        elif n.endswith('rep'):
            print("adding: %s" % n)
            ds = find_downsamples(l)
            ds_list.extend([('%s_%s' % (n, i), d) for i, d in enumerate(ds)])
        else:
            # skip duplicates counted in .rep modules
            # and the .skip modules
            print("skipping: %s" % n)
    return ds_list

def find_rf_model(which_model_netD):
    model = networks.define_patch_D(which_model_netD, init_type=None)
    if 'resnet' in which_model_netD:
        ds = find_downsamples_resnet(model)[::-1]
    elif 'xception' in which_model_netD:
        ds = find_downsamples_xceptionnet(model)[::-1]
    else:
        raise NotImplementedError
    out = 1
    for name, layer in ds:
        k = (layer.kernel_size if isinstance(layer.kernel_size, int) else
             layer.kernel_size[0])
        s = (layer.stride if isinstance(layer.stride, int) else
             layer.stride[0])
        out = rf(out, k, s)
        print('%s, %d' % (name, out))
    return out

def find_rf_patches(which_model_netD, input_size):
    ''' given a patch model name and size of input image,
        calculate the receptive field of each coordinate of the output patch
        returns a dictionary mapping output (h,w) coordinate
        to input h1:h2, w1:w2 slices
    '''
    # modified from https://github.com/rogertrullo/Receptive-Field-in-Pytorch/blob/master/Receptive_Field.ipynb

    # this can be kind of slow, so see if there are saved results first
    import pickle
    rf_cache = os.path.join('utils/rf_cache', '%s_rf_patches_%d.pkl' %
                            (which_model_netD, input_size))
    if os.path.isfile(rf_cache):
        RF = pickle.load(open(rf_cache, 'rb'))
        print("Found existing RF patches")
        return RF

    model = networks.define_patch_D(which_model_netD, init_type=None)

    # the following 2 steps are needed for this method to work
    # (1) replace maxpool layers with avgpool layers
    if hasattr(model, 'maxpool'): # resnet and resnet patch variations
        layer = getattr(model, 'maxpool')
        model.maxpool = nn.AvgPool2d(layer.kernel_size, layer.stride,
                                     layer.padding)
    elif hasattr(model, 'model'): # widenet variations
        if isinstance(model.model[3], torch.nn.MaxPool2d):
            layer = model.model[3]
            model.model[3] = nn.AvgPool2d(layer.kernel_size, layer.stride,
                                          layer.padding)
    elif hasattr(model, 'block1'): # xceptionet variations
        for b in range(1, 13):
            if not hasattr(model, 'block%d' % b):
                break # model was truncated here
            layer = getattr(model, 'block%d' % b)
            if isinstance(layer.rep[-1], torch.nn.MaxPool2d):
                print('here: block%d' % b)
                maxpool = layer.rep[-1]
                layer.rep[-1] = nn.AvgPool2d(maxpool.kernel_size,
                                             maxpool.stride,
                                             maxpool.padding)
    # (2) put in eval mode
    model.eval()

    # makes the gradients a bit more consistent
    for n,m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight = torch.nn.Parameter(torch.ones_like(m.weight))


    img_ = torch.ones((1, 3, input_size, input_size), requires_grad=True)
    print("Input shape: %s" % str(img_.shape))
    out_cnn=model(img_)
    out_shape=out_cnn.size()
    print("Output shape: %s" % str(out_shape))
    RF = OrderedDict()
    for h in tqdm(range(out_shape[2])):
        for w in range(out_shape[3]):
            grad=torch.zeros(out_cnn.size())
            l_tmp = [0, 0, h, w]
            l_tmp = [int(x) for x in l_tmp]
            grad[tuple(l_tmp)]=1
            out_cnn.backward(gradient=grad, retain_graph=True)
            grad_np=img_.grad[0,0].data.numpy()
            idx_nonzeros=np.where(grad_np!=0)
            RF[(h, w)] = (slice(np.min(idx_nonzeros[0]),
                                np.max(idx_nonzeros[0])+1),
                          slice(np.min(idx_nonzeros[1]),
                                np.max(idx_nonzeros[1])+1))
            img_.grad.data.zero_() # zero the gradient

    # save the result to cache
    os.makedirs('utils/rf_cache', exist_ok=True)
    pickle.dump(RF, open(rf_cache, 'wb'))

    return RF

def get_patch_from_img(img, indices, rf, pad_value=0):
    ''' given an image and indices to extract a patch from,
        extracts a patch of size rf by padding the border
        if necessary
    '''

    slice_h, slice_w = indices
    assert(np.ndim(img) == 3) # no batch dim
    assert(img.shape[0] in [1, 3]) # channels first

    padded = np.ones((img.shape[0], rf, rf)) * pad_value
    patch = img[:, slice_h, slice_w]
    if patch.shape == padded.shape:
        return patch
    offset_h = max(0, rf - slice_h.stop)
    offset_w = max(0, rf - slice_w.stop)
    slice_h_offset = slice(offset_h, slice_h.stop - slice_h.start + offset_h)
    slice_w_offset = slice(offset_w, slice_w.stop - slice_w.start + offset_w)
    padded[:, slice_h_offset, slice_w_offset] = patch
    return padded

def find_rf_numerical(which_model_netD, input_size):
    ''' computes the receptive field numerically by finding where
        there are non-zero gradients in the input wrt to an output
        location, should give the same answer as find_rf_model if
        rf < input_size
    '''
    model = networks.define_patch_D(which_model_netD, init_type=None)

    # the following 2 steps are needed for this method to work
    # (1) replace maxpool layers with avgpool layers
    if hasattr(model, 'maxpool'): # resnet and resnet patch variations
        layer = getattr(model, 'maxpool')
        model.maxpool = nn.AvgPool2d(layer.kernel_size, layer.stride,
                                     layer.padding)
    elif hasattr(model, 'model'): # widenet variations
        if isinstance(model.model[3], torch.nn.MaxPool2d):
            layer = model.model[3]
            model.model[3] = nn.AvgPool2d(layer.kernel_size, layer.stride,
                                          layer.padding)
    elif hasattr(model, 'block1'): # xceptionet variations
        for b in range(1, 13):
            if not hasattr(model, 'block%d' % b):
                break # model was truncated here
            layer = getattr(model, 'block%d' % b)
            if isinstance(layer.rep[-1], torch.nn.MaxPool2d):
                print('here: block%d' % b)
                maxpool = layer.rep[-1]
                layer.rep[-1] = nn.AvgPool2d(maxpool.kernel_size,
                                             maxpool.stride,
                                             maxpool.padding)
    # (2) put in eval mode
    model.eval()

    # makes the gradients a bit more consistent
    for n,m in model.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            m.weight = torch.nn.Parameter(torch.ones_like(m.weight))

    img_ = torch.ones((1, 3, input_size, input_size), requires_grad=True)
    print("Input shape: %s" % str(img_.shape))
    out_cnn=model(img_)
    out_shape=out_cnn.size()
    print("Output shape: %s" % str(out_shape))
    assert(len(out_shape) == 4)
    grad=torch.zeros(out_cnn.size())
    l_tmp = [0, 0, out_shape[2] // 2, out_shape[3] // 2]
    l_tmp = [int(x) for x in l_tmp]
    grad[tuple(l_tmp)]=1
    out_cnn.backward(gradient=grad, retain_graph=True)
    grad_np=img_.grad[0,0].data.numpy()
    idx_nonzeros=np.where(grad_np!=0)
    print('RF (h) = %d' % (np.max(idx_nonzeros[0])+1 -
          np.min(idx_nonzeros[0])))
    print('RF (w) = %d' % (np.max(idx_nonzeros[1])+1 -
          np.min(idx_nonzeros[1])))
    img_.grad.data.zero_() # zero the gradient
