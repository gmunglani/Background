# -*- coding: utf-8 -*-
"""
===========================================================
Background subtraction with the DBSCAN clustering algorithm
===========================================================

"""
from __future__ import print_function

print(__doc__)

import numpy as np
import scipy as sp
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import cv2
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from scipy.interpolate import griddata
import math
import pylab

# #############################################################################
move = 8
tile = 16
track = 20
num_images = 100
bleach = np.empty([num_images])
bound = np.empty([num_images])

tif_path = '/home/gm/Documents/Work/Images/Ratio_tubes/TIFF/'

for b in range(num_images):
    print('Image: '+str(b+1))
  #  im = Image.open(tif_path+'_test'+str(b+1)+'.tif')
    im = Image.open(tif_path+'Poster_YC_vid'+str(b+1)+'.tif')
    im2 = np.asarray(im).T
    var = np.empty([3])

    siz = im2.shape
    height = int(siz[1]/move) -1
    width = int(siz[0]/move)-1

    im_median = np.zeros([width,height])
    print(siz)
    for x in range(0, width, 1):
        for y in range(0, height, 1):
            im_test = np.ravel(im2[x*move:x*move+tile,y*move:y*move+tile])
            var = np.vstack((var,np.hstack((sp.stats.moment(im_test,moment=2,axis=0),sp.stats.moment(im_test,moment=3,axis=0),sp.stats.moment(im_test,moment=4,axis=0)))))
            im_median[x,y] = np.median(im2[x*move:x*move+tile,y*move:y*move+tile])

    xgrid = np.arange(0,height)
    ygrid = np.arange(0,width)
    X, Y = np.meshgrid(xgrid, ygrid)
    X1D = np.ravel(X)
    Y1D = np.ravel(Y)

    var = np.delete(var,(0),axis=0)
    varn = np.copy(var)
    varn[:,0] = (varn[:,0]-np.amin(varn[:,0]))/(np.amax(varn[:,0])-np.amin(varn[:,0]))
    varn[:,1] = (varn[:,1]-np.amin(varn[:,1]))/(np.amax(varn[:,1])-np.amin(varn[:,1]))
    varn[:,2] = (varn[:,2]-np.amin(varn[:,2]))/(np.amax(varn[:,2])-np.amin(varn[:,2]))

    db = DBSCAN(eps=0.01, min_samples=100).fit(varn)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels1D = db.labels_
    labels = np.reshape(labels1D,[width,height])

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels1D)) - (1 if -1 in labels1D else 0)

    im_median_mask = np.multiply(im_median,(labels+1))
    im_median_mask1D = np.ravel(im_median_mask)

    XY1D = np.column_stack((X1D,Y1D))
    pos_front = np.where(im_median_mask1D==0)[0]
    XY1D_back = np.delete(XY1D, pos_front, axis=0)
    im_median_mask1D_back = np.delete(im_median_mask1D, pos_front, axis=0)
    XY_interp1D_back = griddata(XY1D_back, im_median_mask1D_back, (X, Y), method='nearest')

    im_left = im_median - XY_interp1D_back
    im3 = np.copy(im2)
    im3.setflags(write=1)

    for x in range(0, width, 1):
        for y in range(0, height, 1):
            im3[x*tile:x*tile + tile, y*tile:y*tile + tile] = np.subtract(im3[x*tile:x*tile+tile,y*tile:y*tile+tile],int(XY_interp1D_back[x,y]))

    high_values_flags = im3 > 65000
    im3[high_values_flags] = 0

    im3 = cv2.GaussianBlur(im3,(3,3),0)
    low_values_flags = im3 < 100
    im3[low_values_flags] = 0

    type = np.argmax(np.hstack((np.sum(im3[:,0]),np.sum(im3[:,-1]),np.sum(im3[0,:]),np.sum(im3[-1,:]))))

    # BACKGROUND SUBTRACTION COMPLETED
    ####################################################################################################
    var_front = np.empty([3])
    im_median_front = np.zeros([width,height])

    for x in range(0, width, 1):
        for y in range(0, height, 1):
            im_test = np.ravel(im3[x*move:x*move+tile,y*move:y*move+tile])
            var_front = np.vstack((var_front,np.hstack((np.mean(im_test),sp.stats.moment(im_test,moment=3,axis=0),sp.stats.moment(im_test,moment=4,axis=0)))))
            im_median_front[x,y] = np.median(im3[x*move:x*move+tile,y*move:y*move+tile])

    var_front = np.delete(var_front,(0),axis=0)

    if (type == 0):
        bound_mask = np.arange(0,width*(height-1),height)
    elif(type == 1):
        bound_mask = np.arange(height-1,width*height-1,height)
    elif(type == 2):
        bound_mask = np.arange(0,height-1,1)
    else:
        bound_mask = np.arange(width*(height-1),width*height-1,1)

    bound_front_mask = bound_mask[np.where(np.in1d(bound_mask,pos_front)*1 == 1)[0]]
    im_median_new1D = np.ravel(im_median_front)
    im_median_bound1D = im_median_new1D[bound_front_mask]
    bound[b] = np.sum(im_median_bound1D)/np.count_nonzero(im_median_bound1D)

    XY1D_front = XY1D[pos_front]
    varn_front = var_front[pos_front,:]
    im_median_front1D = np.ravel(im_median_front)

    varn_front[:,0] = ((varn_front[:,0]-np.amin(varn_front[:,0]))/(np.amax(varn_front[:,0])-np.amin(varn_front[:,0]))) - 1
    varn_front[:,1] = (varn_front[:,1])/(np.amax(varn_front[:,1])-np.amin(varn_front[:,1]))
    varn_front[:,2] = (varn_front[:,2]-np.amin(varn_front[:,2]))/(np.amax(varn_front[:,2])-np.amin(varn_front[:,2]))

    norm_front = np.linalg.norm(varn_front,ord=2,axis=1)
    normpos_front = np.vstack((norm_front, pos_front)).T
    normpos_front_sort = normpos_front[normpos_front[:,0].argsort()]
    pos_front_select = normpos_front_sort[0:track,1].astype(int)
    im_median_front1D_select = im_median_front1D[pos_front_select]
    bleach[b] = np.mean(im_median_front1D_select)

    im_front_select_mask = np.zeros([siz[0],siz[1]])
    for a in range(len(pos_front_select)):
        rowpos = int(pos_front_select[a]/width)
        colpos = pos_front_select[a]%height
        im_front_select_mask[rowpos*move:rowpos*move+tile, colpos*move:colpos*move+tile] = 65535


    # Plots and images
#    print('Estimated number of clusters: %d' % n_clusters_)
#    print('Total number of tiles: %d' % labels1D.size)
#    print('Total number of tiles with signal: %d' % -np.sum(labels))

  #  img1 = Image.fromarray(im3.T)
    #img1.show()

  #  img2 = Image.fromarray(im_front_select_mask.T)
    #img2.show()

  #  fig1 = plt.figure()
  #  ax = fig1.gca(projection='3d')
  #  surf1 = ax.plot_surface(X, Y, im_median, cmap=cm.bwr, linewidth=0, antialiased=False)
  #  ax.set_xlabel('X',labelpad=15)
  #  ax.set_ylabel('Y',labelpad=15)
  #  ax.set_zlabel('Fluorescent Intensity',labelpad=15)
  #  ax.set_zlim(0, 550)
  #  ax.grid(False)

  #  fig2 = plt.figure()
  #  ax = fig2.gca(projection='3d')
  #  surf1 = ax.plot_surface(X, Y, XY_interp1D_back, cmap=cm.bwr, linewidth=0, antialiased=False)
  #  ax.set_xlabel('X',labelpad=15)
  #  ax.set_ylabel('Y',labelpad=15)
  #  ax.set_zlabel('Fluorescent Intensity',labelpad=15)
  #  ax.set_zlim(0, 550)
  #  ax.grid(False)

  #  fig3 = plt.figure()
  #  ax = fig3.gca(projection='3d')
  #  surf1 = ax.plot_surface(X, Y, im_left, cmap=cm.bwr, linewidth=0, antialiased=False)
  #  ax.set_xlabel('X',labelpad=15)
  #  ax.set_ylabel('Y',labelpad=15)
  #  ax.set_zlabel('Fluorescent Intensity',labelpad=15)
  #  ax.set_zlim(0, 550)
  #  ax.grid(False)

  #  fig2 = plt.figure()
  #  ax = fig2.add_subplot(111, projection='3d')
  #  ax.set_xlabel('Mean')
  #  ax.set_ylabel('Skewness')
  #  ax.set_zlabel('Kurtosis')
  #  ax.set_xlim(0, 1)
  #  ax.set_ylim(-0.5, 0.5)
  #  ax.set_zlim( 0, 1)
  #  ax.scatter(varn_front[:,0], varn_front[:,1], varn_front[:,2], c='red')

  #  fig3 = plt.figure()
  #  ax = fig3.add_subplot(111, projection='3d')
  #  ax.set_xlabel('Variance')
  #  ax.set_ylabel('Skewness')
  #  ax.set_zlabel('Kurtosis')
  #  ax.set_xlim(0, 1)
  #  ax.set_ylim(-0.5, 0.5)
  #  ax.set_zlim( 0, 1)
  #  xyz = varn[core_samples_mask]
   # ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:,2], c='red')
  #  xyz2 = varn[~core_samples_mask]
   # ax.scatter(xyz2[:, 0], xyz2[:, 1], xyz2[:,2], c='blue')

 #   plt.show()


    #print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    #print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    #print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    #print("Adjusted Rand Index: %0.3f"
    #      % metrics.adjusted_rand_score(labels_true, labels))
    #print("Adjusted Mutual Information: %0.3f"
    #      % metrics.adjusted_mutual_info_score(labels_true, labels))
    #print("Silhouette Coefficient: %0.3f"
    #      % metrics.silhouette_score(X, labels))

    # #############################################################################
    # Plot result
print('Average intensity of brightest '+str(bleach))
print('Average intensity of boundary '+str(bound))

#pylab.rc('font', family='serif', size=30)

fig4 = plt.figure()
#plt.plot(np.arange(0,num_images),bound, c='red',marker='o')
ax = plt.gca()
plt.plot(np.arange(0,num_images),bleach, c='red',marker='o')
ax.set_xlabel('Time', labelpad=15, fontsize=30)
ax.set_ylabel('Fluorescent Intensity', labelpad=15, fontsize=30)
plt.tick_params(axis='both', which='major', labelsize=20)
ax.grid(False)
plt.show()