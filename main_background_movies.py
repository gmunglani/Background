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
import h5py
import logging
from functions import analysis_plot
from timeit import default_timer as timer

# #############################################################################
# Input parameters
start = 1 # Start number of frames
stop = 10 # End number of frames
specific = [] # Specific frames that need their own eps
eps = [0.002]  # DBSCAN tolerance [higher epsilon = more background] - As low as possible

# Options
analysis = 2 # 1 - Create animation of background subtraction, 2 - Create h5 file, 3 - Create both

# Path to files
inp_path = '/home/gm/Documents/Work/Images/Ratio_tubes' # Path with the input Tiff files
out_path = '/home/gm/Documents/Scripts/MATLAB/Tip_results' # Output path for further processing
fname = 'YC18' # Sample name
val = ['YFP'] # ENSURE THAT THE SIZE OF EPS AND VAL ARE THE SAME

# Create path/folder
work_inp_path = inp_path + '/' + fname
work_out_path = out_path+'/'+fname+'/'
if not os.path.exists(work_out_path):
    os.makedirs(work_out_path)
work_out_path += fname

class frame(object):
    def __init__(self,im_stack,count):
        print('Image: ' + str(count + 1))
        self.im_frame = np.asarray(im_stack[count])
        self.count = count

class stack(frame):
    def __init__(self,work_inp_path,val,eps,*args):
        self.val = val
        self.eps = eps
        self.saver = 0

        if (len(args) == 2):
            self.range = np.arange(args[0]-1,args[1])
            self.type = False
        else:
            specific = args[0]
            specific = [x - 1 for x in specific]
            self.range = specific
            self.type = True

        im_path =  work_inp_path + '_' + self.val + '.tif'
        self.im_stack = pims.TiffStack_pil(im_path)

        self.siz1,self.siz2 = self.im_stack.frame_shape
        window = 40 if self.eps > 0.01 else 80
        self.dim = np.int16(self.siz2/window)
        self.height = np.int16(window)
        self.width = np.int16(self.siz1/self.dim)

        self.X, self.Y = np.int16(np.meshgrid(np.arange(self.height), np.arange(self.width)))
        self.XY = np.column_stack((np.ravel(self.X),np.ravel(self.Y)))

    @classmethod
    def logit(self,work_out_path):
        """ Logging input values """
        logger = logging.getLogger('back')
        hdlr = logging.FileHandler(work_out_path + '_back.log')
        formatter = logging.Formatter('%(asctime)s %(message)s')
        hdlr.setFormatter(formatter)
        logger.addHandler(hdlr)
        logger.setLevel(20)

        self.logger = logger

    def metric_prealloc(self):
        length = len(self.range)
        rows = self.height*self.width
        self.im_medianf = np.empty((self.width,self.height,length),dtype=np.float32)
        self.propf = np.empty((rows,4,length),dtype=np.float32)
        self.maskf = np.empty((rows,length),dtype=np.bool)
        self.labelsf = np.empty((rows,length),dtype=np.int8)
        self.im_backf = np.empty((self.width,self.height,length),dtype=np.int16)
        self.im_framef = np.empty((length,self.siz1,self.siz2),dtype=np.uint16)

    def properties(self,count):
        self.ind = frame(self.im_stack,count)
        tile_prop = np.empty([self.width*self.height,4],dtype=np.float32)
        im_tile = np.reshape(self.ind.im_frame,(np.int32(self.siz1*self.siz2/self.dim),self.dim))

        self.ind.im_tile_split = np.split(im_tile,np.arange(self.dim,im_tile.shape[0],self.dim))

        for i in range(tile_prop.shape[0]):
            im_tile_split_flat = np.ravel(self.ind.im_tile_split[i])
            tile_prop[i,0] = sp.stats.moment(im_tile_split_flat,moment=2,axis=0)
            tile_prop[i,1] = sp.stats.moment(im_tile_split_flat,moment=3,axis=0)
            tile_prop[i,2] = sp.stats.moment(im_tile_split_flat,moment=4,axis=0)
            tile_prop[i,3] = np.median(im_tile_split_flat)

        self.ind.im_median = np.copy(tile_prop[:,3])

        tile_min = np.amin(tile_prop,axis=0)
        tile_ptp = np.ptp(tile_prop,axis=0)

        for j in range(tile_prop.shape[1]):
            tile_prop[:,j] = map(lambda x : (x - tile_min[j])/tile_ptp[j], tile_prop[:,j])

        self.ind.tile_prop = tile_prop

    def clustering(self):
        db = DBSCAN(eps=self.eps, min_samples=int(self.height*1.25)).fit(self.ind.tile_prop)
        self.ind.core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        self.ind.core_samples_mask[db.core_sample_indices_] = True
        self.ind.labels = np.int8(db.labels_)

    def subtraction(self):
        im_median_mask = np.multiply(self.ind.im_median,(self.ind.labels+1))
        pos_front = np.int16(np.where(im_median_mask==0)[0])
        XY_back = np.delete(self.XY, pos_front, axis=0)
        im_median_mask_back = np.delete(im_median_mask, pos_front, axis=0)

        try:
            self.ind.XY_interp_back = np.int16(griddata(XY_back, im_median_mask_back, (self.X, self.Y), method='nearest'))
        except:
            self.ind.XY_interp_back = np.zeros((self.width,self.height))
            self.logger.error("".join(self.val,'_eps: ',str(self.eps),', frame: ',str(ind.count+1)," (eps value too low)"))


        im_frame_split = np.empty((np.int32(self.siz1*self.siz2/self.dim),self.dim),dtype=np.int16)
        for i,j in enumerate(self.ind.XY_interp_back.flat):
            im_frame_split[i*self.dim:(i+1)*self.dim,0:self.dim] = np.subtract(self.ind.im_tile_split[i],j)

        self.ind.im_frame = np.reshape(im_frame_split,(self.siz1,self.siz2))

        low_values_flags = self.ind.im_frame < 0
        self.ind.im_frame[low_values_flags] = 0

        self.ind.im_frame = np.float32(self.ind.im_frame)

    def filter(self):
        filtered = cv2.bilateralFilter(self.ind.im_frame, np.int16(math.ceil(9 * self.siz2 / 1280)), 30, 30)
        self.ind.im_frame = np.uint16(filtered)

    def metrics(self):
        self.im_medianf[:,:,self.saver] = np.reshape(self.ind.im_median,(self.width,self.height))
        self.im_backf[:,:,self.saver] = self.ind.XY_interp_back
        self.im_framef[self.saver,:,:] = self.ind.im_frame
        self.propf[:,:,self.saver] = self.ind.tile_prop
        self.maskf[:,self.saver]  = self.ind.core_samples_mask.tolist()
        self.labelsf[:,self.saver] = self.ind.labels
        self.saver += 1

    def logger_update(self,analysis):
        al = True if analysis is not 2 else False
        if (self.type):
            self.logger.info(self.val + '_eps: ' + str(self.eps) + ', frames: ' + ",".join(
                map(str, [x + 1 for x in self.range])) + ', save: ' + str(al))
        else:
            self.logger.info(self.val + '_eps: ' + str(self.eps) + ', frames: ' + str(self.range[0]+1) + '-' + str(
                self.range[-1]+1) + ', save: ' + str(al))

    def h5(self,work_out_path):
        path = work_out_path + '_back.h5'
        f = h5py.File(path, 'a')
        if (self.type):
            orig = f[self.val]
            for i,j in enumerate(self.range):
                orig[j] = self.im_framef[i]
        else:
            k = self.val in f
            if (k):
                del f[k]
            dst = f.create_dataset(self.val, data=self.im_framef, shape=((len(self.range)), self.siz1, self.siz2),
                                       dtype=np.uint16, compression='gzip')

    def plot(self,work_out_path):
            X1, Y1 = np.int16(np.meshgrid(np.arange(self.siz2), np.arange(self.siz1)))
            analysis_plot(X1, Y1, self, work_out_path)

startt = timer()
for a,b in zip(val,eps):
    if (specific):
        all = stack(work_inp_path,a,b,specific)
    else:
        all = stack(work_inp_path,a,b,start,stop)

    all.logit(work_out_path)
    all.metric_prealloc()

    for count in all.range:
        all.properties(count)
        all.clustering()
        all.subtraction()
        all.filter()
        all.metrics()

    all.logger_update(analysis)

    if (analysis > 1):
        all.h5(work_out_path)

    if (analysis != 2):
        all.plot(work_out_path)

endt = timer()
print(startt-endt)



