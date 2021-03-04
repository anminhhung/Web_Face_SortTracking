from flask import Flask, render_template, Response, request
import cv2 
import time
import numpy as np
import imutils
import threading
import logging
import os

from imutils.video import VideoStream

from utils.parser import get_config
from utils.draw_image import draw_list_bbox_maxmin, draw_bbox_maxmin, write_text

from src.detect.dnn.detect_dnn import detect_face_ssd
from src.tracking.sort_tracking import * 
from src.PCN.PyPCN import *


cfg = get_config()
cfg.merge_from_file('configs/face_detect.yaml')
cfg.merge_from_file('configs/service.yaml')

# Model
PROTOTXT = cfg.FACE_DETECT_MODEL.DNN_PROTOTXT
MODEL = cfg.FACE_DETECT_MODEL.DNN_MODEL
NET_DNN = cv2.dnn.readNetFromCaffe(PROTOTXT, MODEL)

# service
HOST = cfg.SERVICE.HOST
PORT = cfg.SERVICE.PORT
LOG_PATH = cfg.SERVICE.LOG_PATH

# setup sort tracking
SORT_TRACKER = Sort()

OUTPUT_FRAME = None
LOCK = threading.Lock()

# create foler
if not os.path.exists(LOG_PATH):
    os.mkdir(LOG_PATH)

# create logging   
logging.basicConfig(filename=os.path.join(LOG_PATH, str(time.time())+".log"), filemode="w", level=logging.DEBUG, format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
console = logging.StreamHandler()
console.setLevel(logging.ERROR)
logging.getLogger("").addHandler(console)
logger = logging.getLogger(__name__)

app = Flask(__name__, template_folder='templates')

CAP = cv2.VideoCapture(0)

@app.route("/")
def index():
	return render_template("index.html")

def run_tracking():
    # cap = cv2.VideoCapture(0)
    global CAP, OUTPUT_FRAME, LOCK
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

    while True:
        start_time = time.time()

        ret, frame = CAP.read()
        if ret == False:
            break

        ########## MAIN #################
        try:
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

            # cv2.imwrite("frame.jpg", frame)

            print("FPS: ", fps)

        except Exception as e:
            with open("logbug.txt", "a+") as f:
                f.write("{}\n".format(e))
            pass

        with LOCK:
            OUTPUT_FRAME = frame.copy()

        #################################

        # cv2.imshow("Frame", frame)

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

def generate():
    global OUTPUT_FRAME, LOCK

    while True:
        with LOCK:
            if OUTPUT_FRAME is None:
                print("output_frame is None")
                continue
            
            OUTPUT_FRAME = imutils.resize(OUTPUT_FRAME, width=400)
            (flag, encodedImage) = cv2.imencode(".jpg", OUTPUT_FRAME)
            # ensure the frame was successfully encoded
            if not flag:
                continue
            
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feed")
def video_feed():
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

if __name__ == '__main__':
    # start a thread
	# t = threading.Thread(target=run_tracking, args=(32,))
	# t.daemon = True
	# t.start()
    t = threading.Thread(target=run_tracking)
    t.daemon = True
    t.start()
    app.run(host=HOST, port=PORT, debug=True, threaded=True, use_reloader=False)