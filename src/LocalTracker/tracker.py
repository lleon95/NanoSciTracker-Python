import numpy as np
import copy
import cv2 as cv

def computeTrackerRoi(roi):
    x1 = roi[0][0]
    y1 = roi[0][1]
    x2 = roi[1][0]
    y2 = roi[1][1]
    return (x1, y1, x2 - x1, y2 - y1)

class Tracker:
    def __init__(self, colour, timeout=5):
        self.tracker = cv.TrackerKCF_create()
        self.colour = colour
        self.roi = None
        self.timeout = 5
        
    def init(self, frame, roi):
        self.roi = roi
        return self.tracker.init(frame, computeTrackerRoi(roi))
    
    def update(self, frame):
        ok, bbox = self.tracker.update(frame)
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))

        self.roi = (p1, p2)

        if self.timeout == 0:
            return False
        if not ok:
            self.timeout -= 1
        return True
    
def updateTrackers(frame, trackers):
    i = 0
    length = len(trackers)
    
    while i < length:
        state = trackers[i].update(frame)
        if not state:
            length -= 1
            del trackers[i]
        else:
            i += 1
    return trackers



def deployTrackers(colour, bb_list, trackers):
    for i in bb_list:
        tracker = Tracker((0,255,0))
        tracker.init(colour, i)
        trackers.append(tracker)
    return trackers
