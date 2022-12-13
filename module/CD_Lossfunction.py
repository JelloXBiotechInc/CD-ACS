import numpy as np
import copy
from skimage.color import separate_stains, combine_stains

rgb_from_hed = np.array([[0.65, 0.70, 0.29],
                         [0.07, 0.99, 0.11],
                         [0, 0, 0]])

stain0 = rgb_from_hed[0,:]
stain1 = rgb_from_hed[1,:]
rgb_from_hed[2,:] = np.cross(stain0, stain1)

hed_from_rgb = np.linalg.inv(rgb_from_hed)


def colorDeconv(rgb, OD):
    rgb_from_hed = OD
    hed_from_rgb = np.linalg.inv(rgb_from_hed)
    
    stains = separate_stains(rgb, hed_from_rgb)
    
    null = np.zeros_like(stains[:, :, 0])
    h = combine_stains(np.stack((stains[:, :, 0], null, null), axis=-1), rgb_from_hed)
    e = combine_stains(np.stack((null, stains[:, :, 1], null), axis=-1), rgb_from_hed)
    d = combine_stains(np.stack((null, null, stains[:, :, 2]), axis=-1), rgb_from_hed)
    
    return h, e, d

def calculateLoss(h_gt, h_otsu):
    ## mean square error
    loss = np.mean((h_gt.astype(np.float32) - h_otsu.astype(np.float32))**2)
    print('mean-square error:', loss)
    
    return loss