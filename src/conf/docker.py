import os, numpy

DEBUG           = True
SOURCE          = os.getenv('SOURCE') or 0
FALLBACK_SOURCE = '/app/static/test_video.mp4'
FRAMES          = int( os.getenv('FRAMES', 24))
MODEL           = os.getenv("MODEL",    "/app/model/MobileNetSSD_deploy.caffemodel")
PROTOTXT        = os.getenv("PROTOTXT", "/app/model/MobileNetSSD_deploy.prototxt")
FRAMES          = int(os.getenv('FRAMES', 24))
CONFIDENCE      = float(os.getenv('CONFIDENCE' , 0.2))
CLASSES         = [
                   "background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"
                  ]
COLORS          = numpy.random.uniform(0, 255, size=(len(CLASSES), 3))

JSON_ADD_STATUS   = False
