# %% import the package
import cv2
print(cv2.__version__)
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame, Series  # for convenience
import pims
import trackpy as tp
import numpy as np
import math
from tqdm import tqdm
import os
import glob
from skimage.feature import canny
from skimage.draw import line
from skimage.transform import probabilistic_hough_line
from itertools import combinations
# %% glob all file path

fileList = glob.glob('./calibration/*/*.tif')
for fullPath in fileList:
    
    frames = pims.open(fullPath)
    img = np.array(frames[0]); img = (img/256).astype('uint8')
    edges = canny(img, 3,5,20)
    lines = probabilistic_hough_line(edges, threshold=30, line_length=400,
                                    line_gap=200)
    fig = plt.figure()
    for line1, line2 in combinations(lines, 2):
        #p0, p1 = line
        #plt.plot((p0[0], p1[0]), (p0[1], p1[1]))
        x0, y0 = line1[0]; x1, y1 = line1[1]
        a0, b0 = line2[0]; a1, b1 = line2[1]
        theta0 = (y1 - y0)/(x1 - x0); theta1 = (b1 - b0)/(a1 - a0)
        if abs(theta1) > 0.1 or abs(theta0) > 0.1:
            print('line detection wrong')
            break
        distance = abs((b0 + b1)/2 - (y1 + y0)/2)
        if distance > 40:
            break
    plt.imshow(img,cmap='gray')
    plt.plot((x0,x1), (y0,y1))
    plt.plot((a0,a1), (b0,b1))

   
    yMax = np.max(np.array([y1,y0,b1,b0]))
    yMin = np.min(np.array([y1,y0,b1,b0]))
    plt.xlabel('crop with ymin = {}, ymax = {}'.format(yMin, yMax))
    
    fileName = fullPath.split('\\')[-1]
    fileName = fileName.split('_MMStack')[0]
    SavePath = fileName
    if not os.path.exists(SavePath):
        os.mkdir(SavePath)
    
    print(fileName)
    estimateFeatureSize = 5
    minMass = 20000
    
    
    @pims.pipeline
    def cropImage(frame):
        frame[0:yMin,:] = 0
        frame[yMax:,:] = 0    
        return frame
    frames = pims.open(fullPath)
    
    m, n = frames.frame_shape
    t = len(frames)
    temp = np.zeros((m,n,t), dtype = frames.pixel_type)
    for kk in range(t):
        temp[:,:,kk] = np.array(frames[kk])
    median = np.median(temp, axis = 2)
    
    @pims.pipeline
    def removeBack(frame, median):
        frame = frame - median
        return frame
    frames = removeBack(frames, median)
    
    #break
    #frames = cropImage(frames)
    
    fig.savefig(SavePath + '/Line.jpg')
    
    
    
    f = tp.batch(frames, estimateFeatureSize, minmass = minMass)    
    t = tp.link(f, 60, memory=80)
    
    t1 = tp.filter_stubs(t, 10)
    tp.plot_traj(t1)

    Ntrajs = np.max(np.array(t1['particle'])) + 1

    minMoveDistance = 300
    print('there are %s trajectories' % Ntrajs)
    t2 = t1[0:0]
    for i in range(Ntrajs):
        tNew = t1[t1['particle']==i]
        if(len(tNew) < 30):
            continue
        #distData = tp.motion.msd(tNew,1,1,len(tNew))
            #dist = distData.iloc[-1,:]['msd']
        x0 = tNew.iloc[0,:]['x']
        y0 = tNew.iloc[0,:]['y']
        xend = tNew.iloc[-1,:]['x']
        yend = tNew.iloc[-1,:]['y']
        dist = np.sqrt((xend - x0)**2 + (yend - y0)**2)
        print('partile index:' , i ,' traveling distance: ', dist)
        if dist > minMoveDistance:
            t2 = t2.append(tNew)

    k,ax1 = plt.subplots(1, figsize=(60,20))
    #ax.plot(x1,y,'-',linewidth = 2, color='red')
    #ax.plot(x2,y,'-',linewidth = 2, color='red')
    tp.plot_traj(t2)
    ax1.imshow(img, cmap= 'gray')
    k.savefig(SavePath + '/images.jpg')
    t2.to_csv(SavePath + '/pointsData.csv')
    
   
# %%


# %% test
fig = plt.figure()
f = tp.locate(np.array(frames[1318]), estimateFeatureSize)

fig, ax = plt.subplots()
ax.hist(f['mass'], bins=20)
ax.set(xlabel='mass', ylabel='count');

minMass = 20000
fig = plt.figure()
f = tp.locate(frames[1318], estimateFeatureSize, minmass= minMass)
tp.annotate(f, frames[1318])
    
# %%