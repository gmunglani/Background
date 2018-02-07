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
import scipy.io as sio
from sklearn.cluster import DBSCAN
import cv2
import matplotlib.pyplot as plt
import math
import pylab
import pims
import matplotlib.cm as cm
import os
import matplotlib.animation as animation
import mpl_toolkits.mplot3d.axes3d as p3

# #############################################################################
# Input parameters
typ = 0 # 0 - YFP, 1 - CFP
res = 4095 # Resolution in pixels
numi = 235 # Number of images
eps = 0.001 # DBSCAN tolerance [higher epsilon = more background]
decay = np.arange(0,100) # Range for decay calculation

# Options
mat_file = True
decay_plot = True
analysis_plot = False

# Path to files
fname = 'YC11'
inp_path = '/home/gm/Documents/Work/Images/Ratio_tubes'
out_path = '/home/gm/Documents/Scripts/MATLAB/Tip_results'

# Choose between YFP and CFP
if (typ == 0):
    val = 'YFP'
else:
    val = 'CFP'

# Set up paths
path = inp_path+'/'+fname+'_'+val+'.tif'
work_path = out_path+'/'+fname+'/'
im = pims.open(path)

# Create folder if it does not exist
if not os.path.exists(work_path):
    os.makedirs(work_path)

# Initialize variables
bleach = np.empty([numi])
maskf = [0] * numi
n_clustersf = [0] * numi
labels1Df = [0] * numi
signalf = [0] * numi

for image in im:
    # Read in the image and convert to np array
    for count in range(numi):
        print('Image: '+str(count+1))
        im = image[count]
        im2 = np.asarray(im)

        # Width and height of single frame
        siz = im2.shape
        move = int(siz[1]/80)
        tile = int(move*2)
        height = int(siz[1]/move) -1
        width = int(siz[0]/move)-1

        # Initializing arrays
        if (count == 0):
            im_output = np.empty([siz[0], siz[1], numi])
            im_outputf = np.empty([siz[0], siz[1], numi])
            im_medianf = np.empty([width, height, numi])
            im_backf = np.empty([width, height, numi])
            im_unbleachf = np.empty([width, height, numi])
            varnf = np.empty([width*height, 3, numi])


        # BACKGROUND SUBTRACTION
        # Finds the median and higher moments in each window
        var = np.empty([3])
        im_median = np.zeros([width,height])
        for x in range(0, width, 1):
            for y in range(0, height, 1):
                im_test = np.ravel(im2[x*move:x*move+tile,y*move:y*move+tile])
                var = np.vstack((var,np.hstack((sp.stats.moment(im_test,moment=2,axis=0),sp.stats.moment(im_test,moment=3,axis=0),sp.stats.moment(im_test,moment=4,axis=0)))))
                im_median[x,y] = np.median(im2[x*move:x*move+tile,y*move:y*move+tile])

        # Normalize higher moments
        var = np.delete(var,(0),axis=0)
        varn = np.copy(var)
        varn[:,0] = (varn[:,0]-np.amin(varn[:,0]))/(np.amax(varn[:,0])-np.amin(varn[:,0]))
        varn[:,1] = (varn[:,1]-np.amin(varn[:,1]))/(np.amax(varn[:,1])-np.amin(varn[:,1]))
        varn[:,2] = (varn[:,2]-np.amin(varn[:,2]))/(np.amax(varn[:,2])-np.amin(varn[:,2]))

        # DBSCAN clustering with output label matrix of the classifier
        db = DBSCAN(eps=eps, min_samples=100).fit(varn)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels1D = db.labels_
        labels = np.reshape(labels1D,[width,height])

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels1D)) - (1 if -1 in labels1D else 0)

        # Mask by multiplying label matrix with median values (remove areas with no signal)
        im_median_mask = np.multiply(im_median,(labels+1))
        im_median_mask1D = np.ravel(im_median_mask)

        # Creates a grid for visualization
        X, Y = np.meshgrid(np.arange(0,height), np.arange(0,width))
        X1D = np.ravel(X)
        Y1D = np.ravel(Y)

        # Remove positions with signal from grid and then interpolate using the background to estimate the background signal distribution
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
        for x in range(0, width, 1):
            for y in range(0, height, 1):
                im3[x*tile:x*tile + tile, y*tile:y*tile + tile] = np.subtract(im3[x*tile:x*tile+tile,y*tile:y*tile+tile],int(XY_interp1D_back[x,y]))

        # Remove negative values (array is circular)
        high_values_flags = im3 > 65000
        im3[high_values_flags] = 0

        # Updating arrays with properties from a single frame
        im_medianf[:,:,count] = im_median
        im_backf[:,:,count] = XY_interp1D_back
        im_unbleachf[:,:,count] = im_left
        varnf[:,:,count] = varn
        maskf[count] = core_samples_mask.tolist()
        n_clustersf[count] = n_clusters_
        labels1Df[count] = labels1D
        signalf[count] = -np.sum(labels)

        # Blurring and thresholding to remove noise and get a clean image
        im3 = cv2.GaussianBlur(im3,(int(math.ceil(9*siz[1]/1280)),int(math.ceil(9*siz[1]/1280))),0)
        im3_res = im3.astype(np.float)*255.0/res
        im3_res8 = np.array(im3_res.astype(np.uint8))
        ret, im4 = cv2.threshold(im3_res8, 0, 255, cv2.THRESH_OTSU)

        # Unbleached image
        im5 = np.round(np.divide(im4,255)*im3)
        im_output[:,:,count] = im5

        # Bleach calculation
        im5_1D = np.ravel(im5)
        im_corr_pos = im5_1D[np.nonzero(np.ravel(im5_1D))]
        bleach[count] = np.median(im_corr_pos)

# Decay fit and correction
if (decay_plot == True or mat_file == True):
    im_outputf = np.copy(im_output)
    if (decay.shape[0] > 0):
        # Fit decay
        fitt = np.polyfit(decay, bleach[decay], 1)
        dval = np.poly1d(fitt)
        print(fitt)

        # Bleached image
        for a in range(decay[0],decay[1]):
            im_outputf[:,:,a] = np.multiply(im_output[:,:,a],(dval(decay[0])/dval(a)))

# Write mat files
if (mat_file == True):
    sio.savemat(work_path+fname+'_'+val+'.mat', mdict={'arr': im_output.astype(int)})
    sio.savemat(work_path + fname + 'f_' + val + '.mat', mdict={'arr': im_outputf.astype(int)})

# Plot decay of signal
if (decay_plot == True):
    fig1 = plt.figure()
    ax = plt.gca()
    if (decay.shape[0] > 0):
        plt.plot(np.arange(decay[0],numi,1),dval(np.arange(decay[0], numi, 1)), c='red')
    plt.plot(np.arange(1,numi),bleach[np.arange(1,numi)], 'bo')
    ax.set_xlabel(val+' Frame', labelpad=15, fontsize=28)
    ax.set_ylabel('Fluorescent Intensity', labelpad=15, fontsize=28)
    plt.tick_params(axis='both', which='major', labelsize=18)
    ax.grid(False)
    plt.show()

# Create animation of background subtraction
if (analysis_plot == True):
    # Define variables over each frame
    def data(i,X,Y,line):
        ax1.clear()
        line1 = ax1.plot_surface(X,Y,im_medianf[:,:,i],cmap=cm.bwr, linewidth=0, antialiased=False)
        ax1.set_title("{} Frame: {}".format(val,i+1))
        ax1.set_zlim(0, np.amax(im_medianf))
        ax1.grid(False)
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])

        ax2.clear()
        line2 = ax2.plot_surface(X,Y,im_backf[:,:,i],cmap=cm.bwr, linewidth=0, antialiased=False)
        ax2.set_title("Number of Clusters: {}".format(n_clustersf[i]))
        ax2.set_zlim(0, np.amax(im_medianf))
        ax2.grid(False)
        ax2.set_xticklabels([])
        ax2.set_yticklabels([])

        ax3.clear()
        line3 = ax3.plot_surface(X,Y,im_unbleachf[:,:,i],cmap=cm.bwr, linewidth=0, antialiased=False)
        ax3.set_title("Number of Tiles: {}".format(labels1Df[i].size))
        ax3.set_zlim(0, np.amax(im_medianf))
        ax3.grid(False)
        ax3.set_xticklabels([])
        ax3.set_yticklabels([])

        ax4.clear()
        ax4.set_title("Number of Tiles with Signal: {}".format(signalf[i]))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.set_zlim(0, 1)
        ax4.grid(False)
        ax4.set_xlabel('Variance',labelpad=10)
        ax4.set_ylabel('Skewness',labelpad=10)
        ax4.set_zlabel('Kurtosis',labelpad=10)
        xyz = varn[maskf[i]]
        xyz2 = varn[[not i for i in maskf[i]]]
        line4 = ax4.scatter(xyz2[:, 0], xyz2[:, 1], xyz2[:, 2], c='blue')
        line4 = ax4.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c='red', s=80)

        line = [line1, line2, line3, line4]
        return line,

    # Define figures, axis and initialize
    fig2 = plt.figure()
    ax1 = fig2.add_subplot(2,2,1,projection='3d')
    ax2 = fig2.add_subplot(2,2,2,projection='3d')
    ax3 = fig2.add_subplot(2,2,3,projection='3d')
    ax4 = fig2.add_subplot(2,2,4,projection='3d')

    ax1.view_init(elev=15., azim=30.)
    ax2.view_init(elev=15., azim=30.)
    ax3.view_init(elev=15., azim=30.)
    ax4.view_init(elev=30., azim=210.)

    line1 = ax1.plot_surface(X,Y,im_medianf[:,:,0],cmap=cm.bwr)
    line2 = ax2.plot_surface(X,Y,im_backf[:,:,0],cmap=cm.bwr)
    line3 = ax3.plot_surface(X,Y,im_unbleachf[:,:,0],cmap=cm.bwr)
    line4 = ax4.scatter(10, 10, 10, c='red')

    line = [line1, line2, line3, line4]

    # Set up animation
    anim = animation.FuncAnimation(fig2, data, fargs=(X,Y,line),frames=numi, interval=1000, blit=False)

    pylab.rc('font', family='serif', size=10)
    plt.show()

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(extra_args=['-r', '25'])
    anim.save(work_path + fname + '_' + val + '.avi', writer=writer)


print("Fin")


