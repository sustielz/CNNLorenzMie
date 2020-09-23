import numpy as np
from pylorenzmie.analysis import Frame

'''
filter for use with localizer

removes predictions within a certain distance (tol) of each other

input: list of list of dicts (output of Localizer.predict)
output: list of list of dicts, with doubles removed
'''

#### Main function
def nodoubles(frame, tol=5):
    if isinstance(frame, Frame):
        toss = _nodoubles(frame.bboxes, tol)
        frame.remove(toss)
        return frame.bboxes
    else:
        toss = _nodoubles(frame, tol)
        bboxes = frame.copy()
        for i in sorted(toss, reverse=True): 
            bboxes.pop(i)
        return bboxes
    
def _nodoubles(bboxes, tol=5):
        toss = []
        for i, bbox1 in enumerate(bboxes):
            for j, bbox2 in enumerate(bboxes[:i]):
                x1, y1 = bbox1[:2]
                x2, y2 = bbox2[:2]
                dist = np.sqrt((x1-x2)**2 + (y1-y2)**2)
                if dist<tol:
                    toss.append(i)
                    break
        return toss

if __name__=='__main__':
    import json
    preds_file = 'examples/test_yolo_pred.json'

    with open(preds_file, 'r') as f:
        xy_preds = json.load(f)
    
    print('Before:{}'.format(xy_preds))
    #the sample predictions were not close
    #using a ridiculous tolerance for demonstration purposes
    print('After:{}'.format(nodoubles([pred['bbox'] for pred in xy_preds], tol=1000)))
