import cv2
import numpy as np

def draw_list_bbox_maxmin(image, list_bbox, view_id=False, track_id=None):
    for bbox in list_bbox:
        image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    
        if view_id:
            track_id = str(track_id)
            cv2.putText(image, "person_id: " + track_id, (int(bbox[0])+2, int(bbox[1])-1), 0, 1, (255, 0, 0), 1)

    return image

def draw_bbox_maxmin(image, bbox, view_id=False, track_id=None):
    image = cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
    if view_id:
        track_id = str(track_id)
        cv2.putText(image, "person_id: " + track_id, (int(bbox[0])+2, int(bbox[1])-1), 0, 1, (255, 0, 0), 1)

    return image 

def write_text(image, text, x, y):
    image = cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

    return image
