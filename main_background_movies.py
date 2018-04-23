# -*- coding: utf-8 -*-
"""
===========================================================
Background subtraction with the DBSCAN clustering algorithm
===========================================================

"""
from __future__ import print_function, division

print(__doc__)

import numpy as np
import scipy as sp
from scipy.interpolate import griddata
from sklearn.cluster import DBSCAN
import cv2
import math
import pims
import os
from functions import analysis, logit
import h5py
import logging

# #############################################################################
# Input parameters
res = 4095 # Resolution in pixels
start = 1 # Start number of frames
stop = 215 # End number of frames
eps = [0.002, 0.006]  # DBSCAN tolerance [higher epsilon = more background] - As low as possible
specific = [] # Specific frames that need their own eps

# Options
analysis_plot = True # Create animation of background subtraction
h5_file = True # Create h5 file

# Path to files
inp_path = '/home/gm/Documents/Work/Images/Ratio_tubes' # Path with the input Tiff files
out_path = '/home/gm/Documents/Scripts/MATLAB/Tip_results' # Output path for further processing
fname = 'YC11' # Sample name
val = ['YFP','CFP'] # ENSURE THAT THE SIZE OF EPS AND VAL ARE THE SAME

# Create folder if it does not exist
work_path = out_path+'/'+fname+'/'
if not os.path.exists(work_path):
    os.makedirs(work_path)

# Index at 0 instead if 1
start = start - 1

# Setup logging
logger = logit(work_path + fname + '_back.log')

for typ in range(len(val)):
    print(val[typ])
    if specific:
        start = 0
        stop = len(specific)

    path = inp_path + '/' + fname + '_' + val[typ] + '.tif'
    im = pims.TiffStack_pil(path)

    # Initialize variables
    maskf = [0] * (stop-start)
    n_clustersf = [0] * (stop-start)
    labels1Df = [0] * (stop-start)
    signalf = [0] * (stop-start)

    # Read in the image and convert to np array
    for count in range(start,stop):
        if specific:
            print('Image: ' + str(specific[count]))
            im2 = np.asarray(im[specific[count]])
        else:
            print('Image: ' + str(count))
            im2 = np.asarray(im[count])

        # Width and height of single frame
        siz = im2.shape
        window = 40 if eps[typ] > 0.01 else 80
        tile = int(siz[1]/window)
        height = int(siz[1]/tile)
        width = int(siz[0]/tile)

        # Initializing arrays
        if (count == start):
            im_medianf = np.empty([width, height, (stop-start)],dtype=np.float16)
            im_backf = np.empty([width, height, (stop-start)],dtype=np.float16)
            im_unbleachf = np.empty([(stop-start), siz[0], siz[1]],dtype=np.uint16)
            varnf = np.empty([width*height, 4, (stop-start)],dtype=np.float16)

            # Creates a grid for visualization
            X, Y = np.meshgrid(np.arange(0,height), np.arange(0,width))
            X1, Y1 = np.meshgrid(np.arange(0,siz[1]), np.arange(0,siz[0]))
            X1D = np.ravel(X)
            Y1D = np.ravel(Y)

        # BACKGROUND SUBTRACTION
        # Finds the median and higher moments in each window
        var = np.empty([4])
        im_median = np.zeros([width,height])
        for x in range(0, width, 1):
            for y in range(0, height, 1):
                im_test = np.ravel(im2[x*tile:x*tile+tile,y*tile:y*tile+tile])
                var = np.vstack((var,np.hstack((sp.stats.moment(im_test,moment=2,axis=0),sp.stats.moment(im_test,moment=3,axis=0),sp.stats.moment(im_test,moment=4,axis=0),np.median(np.ravel(im_test))))))
                im_median[x,y] = np.median(im2[x*tile:x*tile+tile,y*tile:y*tile+tile])

        # Normalize higher moments
        var = np.delete(var,(0),axis=0)
        varn = np.copy(var)
        varn[:,0] = (varn[:,0]-np.amin(varn[:,0]))/(np.amax(varn[:,0])-np.amin(varn[:,0]))
        varn[:,1] = (varn[:,1]-np.amin(varn[:,1]))/(np.amax(varn[:,1])-np.amin(varn[:,1]))
        varn[:,2] = (varn[:,2]-np.amin(varn[:,2]))/(np.amax(varn[:,2])-np.amin(varn[:,2]))
        varn[:,3] = (varn[:,3]-np.amin(varn[:,3]))/(np.amax(varn[:,3])-np.amin(varn[:,3]))

        # DBSCAN clustering with output label matrix of the classifier
        db = DBSCAN(eps=eps[typ], min_samples=int(height*1.25)).fit(varn)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels1D = db.labels_
        labels = np.reshape(labels1D,[width,height])

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels1D)) - (1 if -1 in labels1D else 0)

        # Mask by multiplying label matrix with median values (re-tile areas with no signal)
        im_median_mask = np.multiply(im_median,(labels+1))
        im_median_mask1D = np.ravel(im_median_mask)

        # Re-tile positions with signal from grid and then interpolate using the background to estimate the background signal distribution
        XY1D = np.column_stack((X1D,Y1D))
        pos_front = np.where(im_median_mask1D==0)[0]
        XY1D_back = np.delete(XY1D, pos_front, axis=0)
        im_median_mask1D_back = np.delete(im_median_mask1D, pos_front, axis=0)

        try:
            XY_interp1D_back = griddata(XY1D_back, im_median_mask1D_back, (X, Y), method='nearest')
        except:
            logger.error(val[typ] + '_eps: ' + str(eps[typ]) + ', frame: ' + str(count+1) + " (eps value too low)")
            raise ValueError('eps value is too low')

        # Signal without background on the pixel level
        im3 = np.copy(im2)
        im3.setflags(write=1)
        im3 = np.uint16(im3)
        for x in range(0, width, 1):
            for y in range(0, height, 1):
                im3[x*tile:x*tile + tile, y*tile:y*tile + tile] = np.subtract(im3[x*tile:x*tile+tile,y*tile:y*tile+tile],int(XY_interp1D_back[x,y]))

        # Re-tile negative values (array is circular)
        high_values_flags = im3 > 65000
        im3[high_values_flags] = 0

        low_values_flags = im3 < 0
        im3[low_values_flags] = 0

        im3 = np.float32(im3)
        im3 = cv2.bilateralFilter(im3,int(math.ceil(9 * siz[1] / 1280)),30,30)
        im3 = np.uint16(im3)

        # Updating arrays with properties from a single frame
        im_medianf[:,:,count-start] = im_median
        im_backf[:,:,count-start] = XY_interp1D_back
        im_unbleachf[count-start,:,:] = im3
        varnf[:,:,count-start] = varn
        maskf[count-start] = core_samples_mask.tolist()
        n_clustersf[count-start] = n_clusters_
        labels1Df[count-start] = labels1D
        signalf[count-start] = -np.sum(labels)

    if (specific):
        logger.info(val[typ] + '_eps: ' + str(eps[typ]) + ', frames: ' + ",".join(map(str, [x+1 for x in specific])) + ', save: ' + str(h5_file))
    else:
        logger.info(val[typ] + '_eps: ' + str(eps[typ]) + ', frames: ' + str(start+1) + '-' + str(stop) + ', save: ' + str(h5_file))

    if (h5_file == True):
        path = work_path + fname + '_back.h5'
        f = h5py.File(path, 'a')
        if specific:
            orig = f[val[typ]]
            for j in range(start,stop):
                orig[specific[j]-1] = im_unbleachf[j]
        else:
            k = val[typ] in f
            if (k):
                del f[k]
            dst = f.create_dataset(val[typ], data=im_unbleachf, shape=((stop - start), siz[0], siz[1]), dtype=np.uint16, compression='gzip')

    if (analysis_plot == True):
        analysis(val[typ], X1, Y1, X, Y, im_medianf, im_backf, im_unbleachf, varnf, maskf, signalf, labels1Df,
                 (stop - start), work_path, fname)
print("Fin")


