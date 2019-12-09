import os, numpy

DEBUG           = True
SOURCE          = os.getenv('SOURCE') or 0

MODEL           = "model/MobileNetSSD_deploy.caffemodel"
PROTOTXT        = "model/MobileNetSSD_deploy.prototxt"
FALLBACK_SOURCE = 'static/test_video.mp4'
FRAMES          = int( os.getenv('FRAMES', 24))
CONFIDENCE      = 0.2
CLASSES         = [
                   "background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"
                  ]
COLORS          = numpy.random.uniform(0, 255, size=(len(CLASSES), 3))

JSON_ADD_STATUS = True
JSON_STATUS_FIELD_NAME = 'code'


