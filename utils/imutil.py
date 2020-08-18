import torch
import numpy as np
from PIL import Image
import cv2
from scipy.ndimage.filters import gaussian_filter

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

# Arrange list of images in a grid with padding
# adapted from: https://colab.research.google.com/github/tensorflow/hub/blob/master/examples/colab/biggan_generation_with_tf_hub.ipynb
def imgrid(imarray_np, cols=5, pad=1):
    if imarray_np.dtype != np.uint8:
        raise ValueError('imgrid input imarray_np must be uint8')
    if imarray_np.shape[1] in [1, 3]:
        # reorder channel dimension
        imarray_np = np.transpose(imarray_np, (0, 2, 3, 1))
    pad = int(pad)
    assert pad >= 0
    cols = int(cols)
    assert cols >= 1
    N, H, W, C = imarray_np.shape
    rows = int(np.ceil(N / float(cols)))
    batch_pad = rows * cols - N
    assert batch_pad >= 0
    post_pad = [batch_pad, pad, pad, 0]
    pad_arg = [[0, p] for p in post_pad]
    imarray_np = np.pad(imarray_np, pad_arg, 'constant', constant_values=255)
    H += pad
    W += pad
    grid = (imarray_np
            .reshape(rows, cols, H, W, C)
            .transpose(0, 2, 1, 3, 4)
            .reshape(rows*H, cols*W, C))
    if pad:
        grid = grid[:-pad, :-pad]
    return grid

def normalize_heatmap(heatmap):
    assert(np.ndim(heatmap) == 2)
    heatmap = heatmap - np.min(heatmap)
    heatmap = heatmap / (np.max(heatmap) +1e-6) # a bit of tolerance in div
    return heatmap

def colorize_heatmap(heatmap, normalize=False):
    if normalize:
        heatmap = normalize_heatmap(heatmap)
    heatmap = np.uint8(255*heatmap)
    colorized = cv2.applyColorMap(heatmap, cv2.COLORMAP_BONE) #cv2.COLORMAP_JET) # COLORMAP_BONE
    colorized = colorized[:, :, ::-1]
    return colorized

def overlay_heatmap(image, heatmap, normalize=False):
    assert(np.ndim(image) == 3)
    assert(np.ndim(heatmap) == 2)
    assert(image.shape[-1] in [1, 3]) # HWC
    if normalize:
        heatmap = normalize_heatmap(heatmap)
    (h, w) = image.shape[0:2]
    overlay = np.uint8(255*heatmap)
    overlay = cv2.resize(overlay, (h, w))
    heatmap = cv2.applyColorMap(overlay, cv2.COLORMAP_JET)
    heatmap = heatmap[:, :, ::-1]
    result = heatmap * 0.5 + image * 0.5
    return np.uint8(result)

def overlay_blur(image, heatmap, normalize=False, blur_sigma=3,
                 add_threshold=False, add_contour=False,
                 threshold=0.5, direction='below', color=[255, 255, 0]):
    heatmap_blurred = gaussian_filter(heatmap, sigma=blur_sigma)
    if normalize:
        heatmap_blurred = normalize_heatmap(heatmap_blurred)
    overlay = overlay_heatmap(image, heatmap_blurred) # already normalized
    if not add_threshold and not add_contour:
        return overlay
    mask = cv2.resize(heatmap_blurred, (image.shape[1], image.shape[0])) >= threshold
    mask = np.expand_dims(mask, axis=-1) # HW1 
    if direction == 'above': # plot overlay red above threshold
        threshold_im = (overlay * np.uint8(mask) +
                        (1 - np.uint8(mask)) * image)
    else:  # plot overlay blue below threshold
        threshold_im = (overlay * (1 - np.uint8(mask)) +
                        np.uint8(mask) * image)
    if not add_contour:
        return threshold_im
    border = border_from_mask(np.squeeze(mask))
    contour_y, contour_x = np.where(border)
    threshold_im[contour_y, contour_x, :] = color
    return threshold_im

def border_from_mask(a):
    out = np.zeros_like(a)
    h = (a[:-1,:] != a[1:,:])
    v = (a[:,:-1] != a[:,1:])
    d = (a[:-1,:-1] != a[1:,1:])
    u = (a[1:,:-1] != a[:-1,1:])
    out[:-1,:-1] |= d
    out[1:,1:] |= d
    out[1:,:-1] |= u
    out[:-1,1:] |= u
    out[:-1,:] |= h
    out[1:,:] |= h
    out[:,:-1] |= v
    out[:,1:] |= v
    out &= ~a
    return out

