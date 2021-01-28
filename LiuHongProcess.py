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
# %% parameter need to set
VideoPath = "./calibration/1um_triangle_pulse_2.avi"
ImagePath = "./tempImage/"
estimateFeatureSize = 19 # must be odd numer
minMass = 200 # calculate the integrate minMass intensity
SavePath = VideoPath.split('/')[-1].split('.avi')[0]
if not os.path.exists(SavePath):
    os.mkdir(SavePath)
# %% read the image and save them as frame image and transfer them from rgb image
# to grayscale
'''
vidcap = cv2.VideoCapture(VideoPath)
success,image = vidcap.read()
image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
count = 0
success = True
while success and count < 3000:
    cv2.imwrite(ImagePath + "frame%d.png" % count, image)     # save frame as PNG file
    success,image = vidcap.read()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # print ('Read a new frame: ', success)
    count += 1
'''
# %%
frames = pims.PyAVReaderIndexed(VideoPath)

@pims.pipeline
def as_grey(frame):
    red = frame[:, :, 0]
    green = frame[:, :, 1]
    blue = frame[:, :, 2]
    return 0.2125 * red + 0.7154 * green + 0.0721 * blue
frames = as_grey(frames)

# %% test with parameter (First run to find the min Mass)
# frames = pims.open(ImagePath + '*.png')
f = tp.locate(frames[1000], estimateFeatureSize)
# plot the mass diagram to check the orginal distribution of the mass histogram
fig, ax = plt.subplots()
ax.hist(f['mass'], bins=20)
ax.set(xlabel='mass', ylabel='count');
# %% second run and ask user input minmass
minMass = int(input('please enter the minMass based on histogram'))
f = tp.locate(frames[1000], estimateFeatureSize, minmass= minMass)
tp.annotate(f, frames[1000]);
# %% check subpixel accuracy
tp.subpx_bias(f)

# %% locate feature in all frames
f = tp.batch(frames, estimateFeatureSize, minmass = minMass)
# %%
t = tp.link(f, 20, memory=5)
# %%
plt.figure()
tp.annotate(t[t['frame'] == 0], frames[0]);
t1 = tp.filter_stubs(t, 10)

# %% check raw figure(optional)
tp.plot_traj(t1)


#%% sort and filter traj to avoid not moving particles

Ntrajs = np.max(np.array(t1['particle'])) + 1
minMoveDistance = 400
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
        
# %%
'''
fig = plt.figure()
plt.imshow(frames[0],cmap = 'gray')
LineData = pd.read_csv('./Line.csv')
h,w = frames[0].shape
def getLine(x,y,theta,h):
    startx = x + y/math.tan(math.radians(theta))
    endx = x - (h - y)/math.tan(math.radians(theta))
    return startx,endx
startx1,endx1 = getLine(LineData['BX'][0], LineData['BY'][0], LineData['Angle'][0], h)
startx2,endx2 = getLine(LineData['BX'][1], LineData['BY'][1], LineData['Angle'][1], h)
x1 = np.array([startx1, endx1]); y =np.array([0,h])
x2 = np.array([startx2, endx2])
plt.plot(x1,y,'-',linewidth = 2, color='red')
plt.plot(x2,y,'-',linewidth = 2, color='red')
'''
# %% calculate line functions
'''
coefficients1 = np.polyfit(x1, y, 1)
coefficients2 = np.polyfit(x2, y, 1)
intensity = []
for i in range(len(frames)):
    ValidIntensity = 0; curFrame = np.array(frames[i])
    for curY in range(h):
        for curX in range(w):
            polynomial1 = np.poly1d(coefficients1)
            polynomial2 = np.poly1d(coefficients2)
            if curY <= polynomial2(curX) and curY >= polynomial1(curX):
                ValidIntensity += curFrame[curY,curX]
                curFrame[curY,curX] = 255
    print(i)
    intensity.append(ValidIntensity)
'''
# %%
'''
intensity = []
for i in tqdm(range(len(frames))):
    curFrame = frames[i]#[:,int(min(x1[0],x2[0])):int(max(x1[1],x2[1]))]
    intensity.append(np.average(np.average(curFrame)))
'''
# %%
'''
f = plt.figure()
plt.plot(intensity)
for i in range(len(frames)):
    if(intensity[i]>100):
        break
plt.title('First flashing frame is ' + str(i) + '. So time is ' + str(i) + '/70.54 s')
f.savefig(SavePath +'/LEDtime.jpg')  
df = DataFrame(intensity, columns=['intensity'])
df.to_csv(SavePath + '/intensity.csv')
'''
# %%

k,ax = plt.subplots(1, figsize=(6,4))
#ax.plot(x1,y,'-',linewidth = 2, color='red')
#ax.plot(x2,y,'-',linewidth = 2, color='red')
tp.plot_traj(t2)

k.savefig(SavePath + '/images.jpg')
t2.to_csv(SavePath + '/pointsData.csv')


# %%
