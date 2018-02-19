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
from scipy.optimize import curve_fit
import scipy.io as sio
from sklearn.cluster import DBSCAN
import cv2
import math
import pylab
import pims
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from skimage.feature import register_translation
import mpl_toolkits.mplot3d.axes3d as p3
from preprocessing_functions import analysis, imshowpair


# #############################################################################
# Input parameters
res = 4095 # Resolution in pixels
numi = 10 # Number of images
eps = [0.001, 0.001]  # DBSCAN tolerance [higher epsilon = more background]
fit = 1 # 0 - Linear, 1 - Exponential
decay = np.arange(0,5) # Range for decay calculation

# Options
mat_file = True
analysis_plot = True
#decay_plot = False

# Path to files
fname = 'YC18'
inp_path = '/Users/htv/Desktop'
out_path = '/Users/htv/Desktop'
val = ['YFP','CFP']

# Create folder if it does not exist
work_path = out_path+'/'+fname+'/'
if not os.path.exists(work_path):
    os.makedirs(work_path)

for typ in range(len(val)):
    print(val[typ])
    path = inp_path + '/' + fname + '_' + val[typ] + '.tif'
    im = pims.TiffStack_pil(path)

    # Initialize variables
    bleach = np.empty([numi])
    maskf = [0] * numi
    n_clustersf = [0] * numi
    labels1Df = [0] * numi
    signalf = [0] * numi

    # Read in the image and convert to np array
    for count in range(numi):
        print('Image: '+str(count+1))
        im2 = np.asarray(im[count])

        # Width and height of single frame
        siz = im2.shape
        tile = int(siz[1]/40)
        height = int(siz[1]/tile)
        width = int(siz[0]/tile)

        # Initializing arrays
        if (count == 0):
            im_medianf = np.empty([width, height, numi])
            im_backf = np.empty([width, height, numi])
            im_unbleachf = np.empty([siz[0], siz[1], numi])
            varnf = np.empty([width*height, 3, numi])

            # Creates a grid for visualization
            X, Y = np.meshgrid(np.arange(0,height), np.arange(0,width))
            X1, Y1 = np.meshgrid(np.arange(0,siz[1]), np.arange(0,siz[0]))
            X1D = np.ravel(X)
            Y1D = np.ravel(Y)

        # BACKGROUND SUBTRACTION
        # Finds the median and higher moments in each window
        var = np.empty([3])
        im_median = np.zeros([width,height])
        for x in range(0, width, 1):
            for y in range(0, height, 1):
                im_test = np.ravel(im2[x*tile:x*tile+tile,y*tile:y*tile+tile])
                var = np.vstack((var,np.hstack((sp.stats.moment(im_test,moment=2,axis=0),sp.stats.moment(im_test,moment=3,axis=0),sp.stats.moment(im_test,moment=4,axis=0)))))
                im_median[x,y] = np.median(im2[x*tile:x*tile+tile,y*tile:y*tile+tile])

        # Normalize higher moments
        var = np.delete(var,(0),axis=0)
        varn = np.copy(var)
        varn[:,0] = (varn[:,0]-np.amin(varn[:,0]))/(np.amax(varn[:,0])-np.amin(varn[:,0]))
        varn[:,1] = (varn[:,1]-np.amin(varn[:,1]))/(np.amax(varn[:,1])-np.amin(varn[:,1]))
        varn[:,2] = (varn[:,2]-np.amin(varn[:,2]))/(np.amax(varn[:,2])-np.amin(varn[:,2]))

        # DBSCAN clustering with output label matrix of the classifier
        db = DBSCAN(eps=eps[typ], min_samples=100).fit(varn)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels1D = db.labels_
        labels = np.reshape(labels1D,[width,height])

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels1D)) - (1 if -1 in labels1D else 0)

        # Mask by multiplying label matrix with median values (retile areas with no signal)
        im_median_mask = np.multiply(im_median,(labels+1))
        im_median_mask1D = np.ravel(im_median_mask)

        # Retile positions with signal from grid and then interpolate using the background to estimate the background signal distribution
        XY1D = np.column_stack((X1D,Y1D))
        pos_front = np.where(im_median_mask1D==0)[0]
        XY1D_back = np.delete(XY1D, pos_front, axis=0)
        im_median_mask1D_back = np.delete(im_median_mask1D, pos_front, axis=0)
        XY_interp1D_back = griddata(XY1D_back, im_median_mask1D_back, (X, Y), method='nearest')

        # Signal without background on the window level
        im_left = im_median - XY_interp1D_back

        # Signal without background on the pixel level
        im3 = np.copy(im2)
        im3.setflags(write=1)
        im3 = cv2.GaussianBlur(im3,(int(math.ceil(9*siz[1]/1280)),int(math.ceil(9*siz[1]/1280))),0)
        im3 = im3.astype(int)
        for x in range(0, width, 1):
            for y in range(0, height, 1):
                im3[x*tile:x*tile + tile, y*tile:y*tile + tile] = np.subtract(im3[x*tile:x*tile+tile,y*tile:y*tile+tile],int(XY_interp1D_back[x,y]))

        # Retile negative values (array is circular)
        high_values_flags = im3 > 65000
        im3[high_values_flags] = 0

        low_values_flags = im3 < 0
        im3[low_values_flags] = 0

        # Updating arrays with properties from a single frame
        im_medianf[:,:,count] = im_median
        im_backf[:,:,count] = XY_interp1D_back
        im_unbleachf[:,:,count] = im3
        varnf[:,:,count] = varn
        maskf[count] = core_samples_mask.tolist()
        n_clustersf[count] = n_clusters_
        labels1Df[count] = labels1D
        signalf[count] = -np.sum(labels)

        # Blurring and thresholding to retile noise and get a clean image
   #     fig = plt.figure()
   #     im3_res = im3.astype(np.float)*255.0/res
   #     im3_res8 = np.array(im3_res.astype(np.uint8))
   #     ret, im4 = cv2.threshold(im3_res8, 0, 255, cv2.THRESH_OTSU)
   #     im5 = np.round(np.divide(im4,255)*im3)
   #     ravelo = np.ravel(im5)
   #     plt.hist(ravelo[np.nonzero(ravelo)], bins=np.arange(-1000,2000,10))
        #plt.show()

        # Unbleached image

        # Bleach calculation
    #    im5_1D = np.ravel(im5)
    #    im_corr_pos = im5_1D[np.nonzero(np.ravel(im5_1D))]
    #    bleach[count] = np.median(im_corr_pos)

    if (analysis_plot == True):
        analysis(val[typ], X1, Y1, X, Y, im_medianf, im_backf, im_unbleachf, varnf, maskf, signalf, labels1Df,
                     numi)

    if (mat_file == True):
        sio.savemat(work_path + fname + '_' + val[typ] + '.mat', mdict={'arr': im_unbleachf.astype(int)})

# Exponential fit function
#def func(x, a, b, c):
#    return a * np.exp(-b * x) + c

# Decay fit and correction
#if (decay_plot == True or mat_file == True):
#    im_outputf = np.copy(im_output)
#    if (decay.shape[0] > 0):
#        # Fit decay
#        if (fit == 0):
#            fitt = np.polyfit(decay, bleach[decay], 1)
#            dval = np.poly1d(fitt)
#        else:
#            fitt, pcov = curve_fit(func, decay, bleach[decay], bounds=([bleach[decay[0]]*0.8, 0, -50], [bleach[decay[0]]*1.2, 0.007, 50]))
#            expf = func(np.arange(decay[0],numi,1), *fitt)

#        print(fitt)

        # Bleached image
#        for a in range(decay[0],numi):
#            if (fit == 0):
#                im_outputf[:,:,a] = np.multiply(im_output[:,:,a],(dval(decay[0])/dval(a)))
#            else:
#                im_outputf[:, :, a] = np.multiply(im_output[:, :, a], (expf[0] / expf[a]))

# Write mat files
#if (mat_file == True):
#    sio.savemat(work_path+fname+'_'+val+'.mat', mdict={'arr': im_output.astype(int)})
#    sio.savemat(work_path + fname + 'f_' + val + '.mat', mdict={'arr': im_outputf.astype(int)})

# Plot decay of signal
#if (decay_plot == True):
#    fig1 = plt.figure()
#    ax = plt.gca()
#    if (decay.shape[0] > 0):
#        if (fit == 0):
#            plt.plot(np.arange(decay[0],numi,1),dval(np.arange(decay[0], numi, 1)), c='red')
#        else:
#            plt.plot(np.arange(decay[0],numi,1),expf, c='red')

#    plt.plot(np.arange(1,numi),bleach[np.arange(1,numi)], 'bo')
#    ax.set_xlabel(val+' Frame', labelpad=15, fontsize=28)
#    ax.set_ylabel('Fluorescent Intensity', labelpad=15, fontsize=28)
#    plt.tick_params(axis='both', which='major', labelsize=18)
#    ax.grid(False)
#    plt.show()


print("Fin")


