import cv2 as cv

'''
Painting tools for tracking
'''
def draw_tracker(frame, tracker):
    # Extract the roi
    p1, p2 = tracker.roi
    # Draw on the frame
    frame = cv.rectangle(frame, p1, p2, tracker.colour, 3, 1)
    return frame
    
def draw_trackers(frame, trackers):
    for i in trackers:
        frame = draw_tracker(frame, i)
    return frame

'''
Painting tools for detection
'''
def draw_detections(frame, bbs, colour=(255,0,0)):
    for i in bbs:
        cv.rectangle(frame, i[0], i[1], colour, 3)
    return frame