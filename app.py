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
from utils.draw_image import draw_list_bbox_maxmin, draw_bbox_maxmin

from src.detect.dnn.detect_dnn import detect_face_ssd
from src.tracking.sort_tracking import * 

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

    while True:
        start_time = time.time()

        ret, frame = CAP.read()

        ########## MAIN #################
        try:
            list_face, list_score, list_classes = detect_face_ssd(frame, NET_DNN)

            if len(list_face) > 0:
                list_face = np.array(list_face)
                # update SORT
                track_bbs_ids = SORT_TRACKER.update(list_face)
                for track in track_bbs_ids:
                    frame = draw_bbox_maxmin(frame, track[:4], True, int(track[4]))
                # cal fps
                fps = round(1.0 / (time.time() - start_time), 2)
                print("fps: ", fps)

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