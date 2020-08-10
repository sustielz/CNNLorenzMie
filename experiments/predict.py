import os
import numpy as np
import cv2, json
import matplotlib.pyplot as plt
from time import time
import pandas as pd

import sys
# sys.path.append('/home/group/python/')
sys.path.append('/home/jackie/Desktop/')

######################## For local computer ##################
#cnn_path = '/home/jackie/Documents/Github/cleanup'
#if cnn_path not in sys.path: sys.path.append(cnn_path)
#    
#plmie_path = '/home/jackie/Documents/Github/cleanup/CNNLorenzTest'
#if plmie_path not in sys.path: sys.path.append(plmie_path)
#############################################################

from pylorenzmie.analysis import Frame, Video

from CNNLorenzMie.Localizer import Localizer
from CNNLorenzMie.Estimator import Estimator
from CNNLorenzMie.crop_feature import crop_frame, est_crop_frame
from CNNLorenzMie.experiments.normalize_image import normalize_video
from CNNLorenzMie.filters import no_edges, nodoubles




video_path = 'CNNLorenzMie/examples/videos/tobot2_3p157hz100.avi'
video_path = video_path if np.size(sys.argv) <= 1 else video_path.replace(video_path.split('/')[-1], sys.argv[1])

batch_est = False




loc = Localizer('holo', weights='_100k')
# keras_head_path = '/home/group/python/CNNLorenzTest/keras_models/predict_stamp_best'
keras_head_path = '/home/jackie/Desktop/CNNLorenzMie/keras_models/predict_stamp_best'
keras_model_path = keras_head_path+'.h5'
keras_config_path = keras_head_path+'.json'
with open(keras_config_path, 'r') as f:
    kconfig = json.load(f)
est = Estimator(model_path=keras_model_path, config_file=kconfig)


myvid = Video(path=video_path)
if not os.path.exists(myvid.path + '/norm_images'):
    print('normalizing...')
    normalize_video(video_path)
else: 
    print('already normalized')
myvid.set_frames()

#### Lists to do all of the estimation at once
est_input_imgs = []
est_input_scales = []
est_input_features = []

t0 = time()
i=0
for frame in myvid.frames[:3]:
# for frame in myvid.frames:
    i += 1
    print('processing frame {} ...'.format(i))
    
    frame.load()
    
    loc.predict(frame)
    
    no_edges(frame)
    nodoubles(frame)
    
    crop_frame(frame)
    
    est_imgs, est_scales, est_feats = est_crop_frame(frame, new_shape=est.pixels)
    if batch_est:        
        est_input_imgs.extend(est_imgs)          #### Add to estimator input stack
        est_input_scales.extend(est_scales)
        est_input_features.extend(est_feats)
    else:
        est.predict(est_imgs, est_scales, est_feats)
        #frame.serialize(save=True)
    
    frame.unload()

if batch_est:
    print('Batch estimating...')
    est.predict(est_input_imgs, est_input_scales, est_input_features)
print('finished in {}'.format(time()-t0))
# t0=time()
# myvid.serialize(save=True, path='predicted.json')
# myvid.serialize(save=True, path='predicted_light.json', omit_feat=['data'])
# print('Serialized in {}'.format(time() - t0))
