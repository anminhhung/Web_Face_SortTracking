import cv2
import numpy as np

def detect_face_ssd(image, net):
    # net = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)
    
    # grab the frame dimensions and convert it to a blob
    (h, w) = image.shape[:2]
    image = cv2.resize(image, (300, 300))
    blob = cv2.dnn.blobFromImage(image, 1.0,
                                (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()
    list_face = []
    list_score = []
    list_classes = [] 

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence < 0.4:
            continue

        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        startX, startY, endX, endY = box.astype("int")
        w = endX - startX
        h = endY - startY

        list_face.append([startX, startY, endX, endY])
        list_score.append(confidence)
        list_classes.append("person")

    return list_face, list_score, list_classes
 