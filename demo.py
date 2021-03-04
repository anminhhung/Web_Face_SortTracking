from src.PCN.PyPCN import *
from src.tracking.sort_tracking import * 
from utils.draw_image import *

SORT_TRACKER = Sort()


def detect_face_PCN():
    SetThreadCount(1)
    path = '/usr/local/share/pcn/'
    detection_model_path = c_str(path + "PCN.caffemodel")
    pcn1_proto = c_str(path + "PCN-1.prototxt")
    pcn2_proto = c_str(path + "PCN-2.prototxt")
    pcn3_proto = c_str(path + "PCN-3.prototxt")
    tracking_model_path = c_str(path + "PCN-Tracking.caffemodel")
    tracking_proto = c_str(path + "PCN-Tracking.prototxt")

    cap = cv2.VideoCapture()
    cap.open("{}://{}:{}@{}:{}".format('rtsp', 'admin', 'Admin123', '113.161.51.163', '554'))
    detector = init_detector(detection_model_path,pcn1_proto,pcn2_proto,pcn3_proto,
            tracking_model_path,tracking_proto, 
            40,1.45,0.5,0.5,0.98,30,0.9,1)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) 
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) 
    fps = cap.get(cv2.CAP_PROP_FPS) 

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        start = time.time()
        face_count = c_int(0)
        raw_data = frame.ctypes.data_as(POINTER(c_ubyte))
        windows = detect_track_faces(detector, raw_data, 
                int(height), int(width),
                pointer(face_count))
        end = time.time()
        list_face = []
        for i in range(face_count.value):
            face_bbox = DrawFace(windows[i],frame)
            DrawPoints(windows[i],frame)
            list_face.append(face_bbox)

        # update SORT
        track_bbs_ids = SORT_TRACKER.update(np.array(list_face))
        for track in track_bbs_ids:
            # frame = draw_bbox_maxmin(frame, track[:4], True, int(track[4]))
            frame = write_text(frame, "person_id: " + str(track[4]), int(track[0]), int(track[1]))

        free_faces(windows)

        fps = int(1 / (end - start))

        cv2.imwrite("frame.jpg", frame)

        print("FPS: ", fps)

detect_face_PCN()