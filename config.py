MODEL_PATH = './models/yolo11x.pt'

VIDEO_PATH = './dataset_for test_task/4.mp4'
NUMBER_OBJECTS = 4

CONFIDENCE_THRESHOLD = 0.85

# banana, apple, orange, scissors, cell phone, spoon, fork
TARGET_CLASSES = [46, 47, 49, 76, 67, 44, 42]


MOTION_THRESHOLD = 1.0
WITH_ROI = False

# SORT SETTINGS
MAX_AGE = 40
MIN_HINTS = 2
IOU_THRESHOLD = 0.25