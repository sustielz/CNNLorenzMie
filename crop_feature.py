import numpy as np
from pylorenzmie.analysis import Feature, Frame, Video
from pylorenzmie.theory import LMHologram
from pylorenzmie.theory import coordinates

'''  Methods for cropping images and returning outputs formatted for use by CNNLorenzMie/pylorenzmie objects
Methods:
    fit_center : bbox (4-tuple) (x, y, w, h)
         
'''
#### Adjust the bbox to fit onto an integer grid within the boundary image. 
def crop_center(image, bbox, square=False):            
    (x, y, w, h) = bbox                        
    w = int(np.round(w))                       
    h = int(np.round(h))                       
    x = np.round(x) if w % 2 == 0 else np.floor(x) + 0.5    #### Round x to nearest integer (w even) or half-integer (w odd)
    y = np.round(y) if h % 2 == 0 else np.floor(y) + 0.5     
    (ymax, xmax) = image.shape[:2] or (y+h, x+w)
    w = 2*min(w/2, x, xmax-x)
    h = 2*min(h/2, y, ymax-y)
    if w < 0 or h < 0:
        print('Error: point ({}, {}) is outside of image of shape {}'.format(x, y, list(image.shape)))
        return None, None
    if square and (w is not h):
        return crop_center(image, (x, y, min(w, h), min(w, h)), square=False)
    else:
        (xbot, ybot, w, h) = tuple(map( lambda i: int(np.round(i)), (x - w/2., y - h/2., w, h) ))  #### Get corner, and convert everything to integer before cropping
        image = image[:, :, 0]             #### Convert to grayscale
        return image[ybot:ybot+h, xbot:xbot+w], (xbot, ybot)

#### Square-fiy bbox, use it to crop image, and save data to feature. 
#### Makes sure Feature has a Model, and then set its coordinates. This prepares the Feature to run optimize.
def crop_features(image, bboxes, feats=[]):
    shape = np.shape(bboxes)    #### Check that bboxes has correct dimensions
    if len(shape) != 2:    
        print('error - bboxes must be a list of tuples or lists')
    elif shape[1] is not 4:
        print('error - elements of bbox must have 4 elements')
        return
    if len(feats) != shape[0]:  #### Make sure feature list has correct size
        if len(feats) != 0:
            print('warning feats ignored: number of features does not match number of bboxes')
        feats = [Feature() for bbox in bboxes]
    for i, feat in enumerate(feats):  #### Read information into features
        (x, y, w, h) = bboxes[i]
        ext = max(int(w), int(h))
        data, corner = crop_center(image, (x, y, ext, ext), square=True)
        feat.data = np.array(data)
        if feat.model is None:
            feat.model = LMHologram()
        feat.model.coordinates = coordinates(shape=(ext, ext), corner=corner)
        feat.model.particle.x_p = x
        feat.model.particle.y_p = y
    return feats

# #### For all of the frame's features that have a bbox, crop-feature them. If all=False, ignore previously-croped features.
# #### Ensure all features have a Model, and if they don't have an instrument, set it. 
def crop_frame(frame, all=False):
    bboxes = []
    feats = []
    for i, feat in enumerate(frame.features):
        if frame.bboxes[i] is not None:
            if all or feat.data is None:
                bboxes.append(frame.bboxes[i])
                feats.append(feat)
    crop_features(frame.image, bboxes, feats)
    ins = frame.instrument
    if ins is not None:
        for feat in feats:
            feat.model.instrument = ins
    
def crop_video(video, all=False):
    if isinstance(video, Video):
        video = video.frames
    if isinstance(video, list):
        for frame in video:
            crop_frame(frame, all=False)
    else:
        print('Warning - video must be a Video or a list of Frames')

            
            
''' An intermediate cropping function for the Estimator. Scales up the bbox and then decimates to achieve the correct shape '''
def est_crop(image, bboxes, feats=[], new_shape=(201, 201)):
    if not isinstance(image, np.ndarray):
        print('error - cannot crop image of type {}'.format(type(image)))
        return [], [], []
        
    est_imgs = []
    est_scales = []
    est_feats = []
    for i, bbox in enumerate(bboxes):
        shape = np.shape(bbox)
        if len(np.shape(bbox)) != 1 or np.shape(bbox)[0] != 4:
            print("error - {} is not a valid bbox".format(bbox))
            return [], [], []
    
        (x, y, w, h) = bbox
        ext = max(int(w), int(h))
    #         scale = max(1, int(np.floor(ext/new_shape[0]) + 1) ) 
        scale =  1 + int(np.floor( ext/new_shape[0] ))
#         print(scale)
        (w, h) = np.multiply(new_shape, scale)
        crop, _ = crop_center(image, (x, y, w, h), square=False)
        est_img = crop[::scale, ::scale]
#         print(np.shape(est_img))
        if np.shape(est_img)[0] < new_shape[0] or np.shape(est_img)[1] < new_shape[1]:
            print('warning: unable to resize bbox {} into estimator shape {}'.format(bbox, new_shape))
            continue
#         print(np.shape(est_img))
#         print('hi')
        est_imgs.append(est_img)
        est_scales.append(scale)
        if len(feats) > i: est_feats.append(feats[i])
    return est_imgs, est_scales, est_feats

def est_crop_frame(frame, new_shape=(201, 201)):
    bboxes = []
    feats = []
    for i, feat in enumerate(frame.features):
        if frame.bboxes[i] is not None:
            bboxes.append(frame.bboxes[i])
            feats.append(feat)
            if feat.model is None:
                feat.model = LMHologram()
    est_imgs, est_scales, est_feats = est_crop(frame.image, bboxes, feats=feats, new_shape=new_shape)
    return est_imgs, est_scales, est_feats

def est_crop_video(video, new_shape=(201, 201)):
    if isinstance(video, Video):
        video = video.frames
    if isinstance(video, list):
        est_input_imgs = []
        est_input_scales = []
        est_input_features = []
        for i, img in enumerate(img_list):
            est_imgs, est_scales, est_features = est_crop_frame(frame, new_shape=new_shape)
            est_input_imgs.extend(est_imgs)
            est_input_scales.extend(est_scales)
            est_input_features.extend(est_features)
        return est_input_imgs, est_input_scales, est_input_features
    else:
        print('Warning - video must be a Video or a list of Frames')




            
# '''
# Cropping function meant for intermediate use in EndtoEnd object

# If you're looking to use cropping as a standalone function, chances are crop.py will be more useful to you.
# # 
# NOTE: this function is left in for compatability with the old CNNLorenzMie'''
# def crop_feature(img_list=[], xy_preds=[], new_shape=(201, 201)):
#     frame_list = []
#     est_input_img = []
#     est_input_scale = []
#     for i, img in enumerate(img_list):
#         bboxes = [pred['bbox'] for pred in xy_preds[i]]
# #         frame_list.append( [crop_feature(img, bbox) for bbox in bboxes] 
#         frame = []
#         est_img_list = []
#         est_scale_list = []
#         for bbox in bboxes:
#             frame.append(crop_feature(img, bbox))
#             est_img, est_scale = crop_toShape(img, bbox)
#             est_img_list.append(est_img)
#             est_scale_list.append(est_scale)
#         frame_list.append(frame)
#         est_input_img.append(est_img_list)
#         est_input_scale.append(est_input_scale)
#     return frame_list, est_input_img, est_input_scale
            

    
 
                
                   
    



if __name__ == '__main__':
    from matplotlib import pyplot as plt
#     def try_fit(bbox, imshape, square):
#         s = 'a square ' if square else ''
#         print('trying to fit {}bbox {} into image dimensions {}...'.format(s, bbox, imshape))
#         print('returned bbox {}'.format(fit_center(bbox, imshape, square=square)))
#         print()
    
#     print("Let's test fit_center:")
#     try_fit( (31., 55., 5, 5), (100, 200), False)
#     try_fit( (31., 55., 5, 5), (100, 200), True)
#     try_fit( (5.1, 4.3, 20, 20), (100, 200), False)
#     try_fit( (5.1, 4.3, 20, 20), (100, 200), True)
#     try_fit( (182.6, 3.2, 50, 50), (100, 200), False)
#     try_fit( (182.6, 3.2, 50, 50), (100, 200), True)
#     try_fit( (50, 199.5, 35, 35), (100, 200), False)
#     try_fit( (50, 199.5, 35, 35), (100, 200), True)    
#     print()
#     print()

    #### testing Frame object ####
    print("Let's load a Frame that's been Localized...")
    frame_path = 'examples/exp/exp010_frames/frame0057.json'
    frame = Frame(info=frame_path)
    
    print('Before crop_frame, the Frame is:')
    df = frame.to_df()
    print(df)
    frame.show()
    print('with columns: {}'.format(list(df.columns)))
    print('cropping...')
    crop_frame(frame)
    print('finished cropping. Now, the Frame is:')
    df = frame.to_df()
    print(df)
    print('with columns: {}'.format(list(df.columns)))
    print()   
    
    print('Shallow serializing...')
    frame.serialize(path='examples/exp/exp010_cropped', omit_feat=['data'])
    print('done. Deep serializing...')
    frame.serialize(path='examples/exp/exp010_cropped_deep')
    print('done.')
    print()
    
    image = frame.image
    imshape = np.shape(image)
#     print('image shape is {}'.format(imshape))
#     plt.imshow(image, cmap='gray')
#     plt.show()
    for i, feat in enumerate(frame.features):
        print('Feature ', i+1)
        print('(x,y) = ', feat.model.particle.r_p[:2])
        print('fitted bbox = {}'.format(fit_center(frame.bboxes[i], imshape=imshape, square=True)) )
        data = feat.data
        print('data shape is {}'.format(np.shape(data)) )
        data = data/255.
        plt.imshow(data, cmap='gray')
        plt.show()
 

