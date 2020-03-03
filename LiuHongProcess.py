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
# %% parameter need to set
VideoPath = "./SPACE50_2HZ&2VPP.avi"
ImagePath = "./tempImage/"
estimateFeatureSize = 11 # must be odd numer
minMass = 200 # calculate the integrate minMass intensity

# %% read the image and save them as frame image and transfer them from rgb image
# to grayscale
vidcap = cv2.VideoCapture(VideoPath)
success,image = vidcap.read()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
count = 0
success = True
while success:
    cv2.imwrite(ImagePath + "frame%d.png" % count, image)     # save frame as PNG file
    success,image = vidcap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print ('Read a new frame: ', success)
    count += 1

# %% test with parameter (First run to find the min Mass)
frames = pims.open(ImagePath + '*.png')
f = tp.locate(frames[0], estimateFeatureSize)
# plot the mass diagram to check the orginal distribution of the mass histogram
fig, ax = plt.subplots()
ax.hist(f['mass'], bins=20)
ax.set(xlabel='mass', ylabel='count');
# %% second run and ask user input minmass
minMass = int(input('please enter the minMass based on histogram'))
f = tp.locate(frames[0], estimateFeatureSize, minmass= minMass)
tp.annotate(f, frames[0]);
# %% check subpixel accuracy
tp.subpx_bias(f)

# %% locate feature in all frames
tp.quiet()
f = tp.batch(frames, estimateFeatureSize, minmass = minMass)
t = tp.link(f, 20, memory=1)
plt.figure()
tp.annotate(t[t['frame'] == 0], frames[0]);
t1 = tp.filter_stubs(t, 300)



#%% sort and filter traj to avoid not moving particles
Ntrajs = np.max(np.array(t1['particle'])) + 1
minMoveDistance = 1000
print('there are %s trajectories' % Ntrajs)
t2 = t1[0:0]
for i in range(Ntrajs):
    tNew = t1[t1['particle']==i]
    if(len(tNew) < 300):
        continue
    distData = tp.motion.msd(tNew,1,1,len(tNew))
    dist = distData.iloc[-1,:]['msd']
    print('partile index:' , i ,' traveling distance: ', dist)
    if dist > minMoveDistance:
        t2 = t2.append(tNew)


h = plt.figure(1, figsize=(6,12))
tp.plot_traj(t2)
h.savefig('images.jpg')
t2.to_csv('./pointsData.csv')
