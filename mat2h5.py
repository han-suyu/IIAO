# -*- coding: <encoding name> -*-
"""
data processing.
"""
from __future__ import print_function, division
import h5py
import scipy.io as io
import numpy as np
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import scipy.spatial
import os
import cv2
from scipy.ndimage.filters import gaussian_filter
# %matplotlib inline

def generate_density(f):
    # root directory
    base = './datasets/jhu_crowd_v2.0/' +f +'/'
 

    h5_folder = base+'density/'
    # shutil.rmtree(mat_folder)
    if not os.path.isdir(h5_folder):
        os.mkdir(h5_folder)


    check_folder = base+'check_density/'
    # shutil.rmtree(mat_folder)
    if not os.path.isdir(check_folder):
        os.mkdir(check_folder)
 




    img_paths = os.listdir(base + 'images')
    img_paths.sort()



    # generate density map
    for name in img_paths:
        if name.endswith('.jpg'):
            print('solving... ',name)
            mat = io.loadmat(base+'mats/'+name.replace('.jpg', '.mat'))
            img = plt.imread(base+'images/'+name)


            k = np.zeros((img.shape[0], img.shape[1]))
            gt = mat["annPoints"]
            for i in range(0, len(gt)):
                if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                    k[int(gt[i][1]), int(gt[i][0])] = 1
            k = gaussian_filter_density(k)


            assert img.shape[0]==k.shape[0] and img.shape[1]==k.shape[1]
        
            with h5py.File(h5_folder+name.replace('.jpg', '.h5'), 'w') as hf:
                hf['density'] = k
    

            heatmapshow = None
            heatmapshow = cv2.normalize(k, heatmapshow, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            heatmapshow = cv2.applyColorMap(heatmapshow, cv2.COLORMAP_JET)
            cv2.imwrite(check_folder+name,heatmapshow)
            #print('\n')
  
    # print("Over!!!")


################################################################################
# internal function.
################################################################################
def gaussian_filter_density(gt):
    im_density = np.zeros(gt.shape, dtype = np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return im_density

    points = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))

    h, w = gt.shape[0:2]
    f_sz=15
    sigma=4
    
    # iterate over all the points
    for j in range(len(points)):
        # create the gaussian kernel
        H = matlab_style_gauss2D((f_sz, f_sz), sigma)
        # limit the bound
        x = np.minimum(w, np.maximum(1, np.abs(np.int32(np.floor(points[j, 0])))))
        y = np.minimum(h, np.maximum(1, np.abs(np.int32(np.floor(points[j, 1])))))
        if x > w or y > h:
            continue
        # get the rect around each head
        x1 = x - np.int32(np.floor(f_sz / 2))
        y1 = y - np.int32(np.floor(f_sz / 2))
        x2 = x + np.int32(np.floor(f_sz / 2))
        y2 = y + np.int32(np.floor(f_sz / 2))
        dx1 = 0
        dy1 = 0
        dx2 = 0
        dy2 = 0
        change_H = False
        if x1 < 1:
            dx1 = np.abs(x1) + 1
            x1 = 1
            change_H = True
        if y1 < 1:
            dy1 = np.abs(y1) + 1
            y1 = 1
            change_H = True
        if x2 > w:
            dx2 = x2 - w
            x2 = w
            change_H = True
        if y2 > h:
            dy2 = y2 - h
            y2 = h
            change_H = True
        x1h = 1 + dx1
        y1h = 1 + dy1
        x2h = f_sz - dx2
        y2h = f_sz - dy2
        if change_H: 
            H = matlab_style_gauss2D((y2h - y1h + 1, x2h - x1h + 1), sigma)
        # attach the gaussian kernel to the rect of this head
        im_density[y1 - 1:y2, x1 - 1:x2] = im_density[y1 - 1:y2, x1 - 1:x2] + H
    return im_density


  

def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h
################################################################################
# main function
#This part of the code is borrowed from https://github.com/TencentYoutuResearch/CrowdCounting-SASNet/blob/main/prepare_dataset.py
################################################################################
if __name__ == '__main__':


    folder = ['train','val','test']

    for f in folder:
        generate_density(f)
        print('******************************************************')
        print('Part  '+f+'  completed!!!')
        print('******************************************************')

   
   
