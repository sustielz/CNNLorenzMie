import numpy as np
from pylorenzmie.analysis import Frame

'''
filter for use with localizer

removes predictions within a certain distance (tol) of each other

input: list of list of dicts (output of Localizer.predict)
output: list of list of dicts, with doubles removed
'''
#### Main function
def no_edges(frame, tol=200, image_shape=(1280,1024)):
    if isinstance(frame, Frame):
        if image_shape is None: 
            image_shape = frame.image.shape
        toss = _no_edges(frame.bboxes, tol, image_shape)
        frame.remove(toss)
        return frame.bboxes
    else:
        toss = _no_edges(frame, tol, image_shape)
        bboxes = frame.copy()
        for i in sorted(toss, reverse=True): 
            bboxes.pop(i)
        return bboxes
    
#### Return indices to be filtered
def _no_edges(bboxes, tol, image_shape):
    minwidth = np.min(image_shape)
    if tol < 0 or tol > minwidth/2:
        print('Invalid tolerance for this frame size')
        return None
    xmin, ymin = (tol, tol)
    xmax, ymax = np.subtract(image_shape, (tol, tol))

    toss = []
    for i, bbox in enumerate(bboxes):
        if bbox is not None and (bbox[0]<xmin or bbox[0]>xmax or bbox[1]<ymin or bbox[1]>ymax):
            toss.append(i)
    return toss

if __name__=='__main__':
    import json
    preds_file = '../examples/test_yolo_pred.json'

    with open(preds_file, 'r') as f:
        xy_preds = json.load(f)
    print('Before:{}'.format(xy_preds))

    #the sample predictions were not close
    #using a ridiculous tolerance for demonstration purposes
    print('After:{}'.format(no_edges([pred['bbox'] for pred in xy_preds], tol=325)))
