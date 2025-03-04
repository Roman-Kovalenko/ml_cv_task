import cv2
import time
from numpy import mean, std
from ultralytics import YOLO
from sort import Sort
from config import MODEL_PATH, VIDEO_PATH, MOTION_THRESHOLD, WITH_ROI, MAX_AGE, MIN_HINTS, IOU_THRESHOLD, NUMBER_OBJECTS
from utils import warmup_yolo_model, resize_image_proportionally, get_tracked_objects, draw_tracked_objects, crop_image, BestFrameSaver


if __name__ == '__main__':
    frame_processing_time = []
    current_tracked_objects = []

    yolo_model = YOLO(MODEL_PATH)
    warmup_yolo_model(model=yolo_model)

    frame_saver = BestFrameSaver(number_objects=NUMBER_OBJECTS)

    sort_tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HINTS, iou_threshold=IOU_THRESHOLD)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print("Не удалось открыть видео")
        exit()

    success, previous_frame = cap.read()
    if not success:
        print("Не удалось прочитать первый кадр")
        exit()
    previous_frame = resize_image_proportionally(image=previous_frame)

    if WITH_ROI:
        user_roi = cv2.selectROI("select the area", previous_frame)
        previous_frame = crop_image(image=previous_frame, roi=user_roi)
        cv2.destroyAllWindows()

    while cap.isOpened():
        start_time = time.time()
        success, current_frame = cap.read()
        if not success:
            print("Не удалось прочитать следующий кадр")
            break
        current_frame = resize_image_proportionally(image=current_frame)
        if WITH_ROI:
            current_frame = crop_image(image=current_frame, roi=user_roi)

        diff = cv2.absdiff(previous_frame, current_frame)
        motion = cv2.mean(diff)[0]

        if motion < MOTION_THRESHOLD:
            detection_enabled = False
        else:
            detection_enabled = True

        if detection_enabled:
            current_tracked_objects = get_tracked_objects(model=yolo_model, tracker=sort_tracker, frame=current_frame)
        end_time = time.time()
        processing_time = end_time - start_time
        frame_processing_time.append(processing_time)
        status = "включена" if detection_enabled else "выключена"
        print(f"Время обработки кадра: {processing_time:.4f} сек (Детекция: {status})")


        previous_frame = current_frame.copy()

        current_frame = draw_tracked_objects(frame=current_frame, tracked_objects=current_tracked_objects, detection_type=detection_enabled)

        frame_saver.checker(tracked_objects=current_tracked_objects, current_time_ms=cap.get(cv2.CAP_PROP_POS_MSEC), current_frame=current_frame)

        cv2.imshow("YOLO Inference", current_frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    
    print(f'Среднее време обработки кадра - {mean(frame_processing_time):.4f} +/- {std(frame_processing_time):.4f}')
    frame_saver.save()

    cap.release()
    cv2.destroyAllWindows()
