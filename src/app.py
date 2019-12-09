#!/usr/bin/env python
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import logging
import numpy as np
from uptime import uptime
from imutils.video import FileVideoStream
from flask_json import json_response

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
application = Flask(__name__)
application.config.from_pyfile(f"conf/{application.config['ENV']}.py")

outputFrame = None
lock = threading.Lock()
application.logger.info("Loading prototxt: " + application.config["PROTOTXT"])
application.logger.info("Loading model: " + application.config["MODEL"])
net = cv2.dnn.readNetFromCaffe(application.config["PROTOTXT"], application.config["MODEL"])

time.sleep(2)
# #vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=application.config["SOURCE"]).start()
time.sleep(2)

def detect_object(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock

    # initialize the motion detector and the total number of frames
    # read thus far
    total = 0
    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = vs.read()
        time.sleep(0.1)

        if frame is None:
            print("INFO: unable to connect to a webcam, using file stream instead")
            vs = FileVideoStream("static/test_video.mp4").start()
            frame = vs.read()
            continue
        frame = imutils.resize(frame, width=400)
        
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
            0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()

        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > application.config["CONFIDENCE"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                classes = application.config["CLASSES"]
                colors  = application.config["COLORS"]

                label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), colors[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

        # acquire the lock, set the output frame, and release the lock
        with lock:
            outputFrame = frame.copy()


def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock

    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue

            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)

            # ensure the frame was successfully encoded
            if not flag:
                continue

        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
            bytearray(encodedImage) + b'\r\n')

@application.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")


@application.route('/')
def index():
    # return the rendered template
    return render_template("index.html")

@application.route('/status')
def root():
    return json_response(
        200,
        status="ok",
        uptime=uptime(),
    )

t = threading.Thread(target=detect_object, args=(application.config["FRAMES"],))
t.daemon = True
t.start()

if __name__ == "__main__":
    application.run(
      host=application.config['HOST'], 
      port=application.config['PORT'], 
      threaded=True,
    )
    # release the video stream pointer
    vs.stop()
