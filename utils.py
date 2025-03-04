import cv2
import torch
import argparse
import numpy as np
from ultralytics.engine.model import Model
from sort import Sort
from config import TARGET_CLASSES, CONFIDENCE_THRESHOLD, VIDEO_PATH


class BestFrameSaver:
    def __init__(self, number_objects: int = 4) -> None:
        self.number_objects = number_objects
        self.best_score = 0.0
        self.best_time_ms = 0
        self.image_to_save = None

    def checker(self, tracked_objects: np.ndarray, current_time_ms: int, current_frame: np.ndarray) -> None:
        if len(tracked_objects) == self.number_objects:
            current_score = np.mean(tracked_objects[:, 5])
            if current_score > self.best_score:
                self.best_score = current_score
                self.best_time_ms = current_time_ms
                self.image_to_save = current_frame

    def save(self) -> None:
        if self.image_to_save is not None:
            name = f"best_frame_{int(self.best_time_ms)}.jpg"
            print(f'Лучшая средняя точность - {self.best_score:.3f} | Файл - {name} сохранён')
            cv2.imwrite(name, self.image_to_save)
        else:
            print('не удалось найти нужный кадр.')

def resize_image_proportionally(image: np.ndarray, scale:float=0.6) -> np.ndarray:
    height, width = image.shape[:2]
    new_width = int(width * scale)
    new_height = int(height * scale)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_NEAREST)


def crop_image(image: np.ndarray, roi: list) -> np.ndarray:
    return image[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
    

def warmup_yolo_model(model: Model, input_size:list[int,int]=(640, 640), num_runs:int=5, device:str='cuda') -> None:
    dummy_input = torch.rand(1, 3, *input_size).to(device)
    print("Разогрев модели ...")
    for _ in range(num_runs):
        _ = model(dummy_input, verbose=False)
    print("Разогрев завершён.")


def get_tracked_objects(model: Model, tracker: Sort, frame: np.ndarray) -> np.array:
    results = model(frame, verbose=False)[0]

    detections = []

    for det in results.boxes.data:
        x1, y1, x2, y2, conf, cls = det.cpu().numpy()
        if int(cls) in TARGET_CLASSES and conf > CONFIDENCE_THRESHOLD:
            detections.append([int(x1), int(y1), int(x2), int(y2), conf])
    
    if detections:
        detections = np.array(detections)
        tracker.update(detections)
    else:
        tracker.update()

    tracks = []
    for cur_tracker in tracker.trackers:
        tracked_bbox = tuple(cur_tracker.get_state()[0].astype(int))
        tracked_id = cur_tracker.id + 1
        score = cur_tracker.score
        tracks.append(np.concatenate((tracked_bbox,[tracked_id],[score])).reshape(1,-1))

    if(len(tracks)>0):
        return np.concatenate(tracks)
    return np.empty((0,6))


def get_color(track_id: int) -> tuple[int, int, int]:
    np.random.seed(track_id)
    return tuple(np.random.randint(100, 255, size=3).tolist())


def draw_tracked_objects(frame: np.ndarray, tracked_objects: list, detection_type: bool) -> np.ndarray:
    for obj in tracked_objects:
        x1, y1, x2, y2, track_id = map(int, obj[:5])
        cur_score = obj[5]

        color_black = (0, 0, 0)
        color_red = (0, 0, 200)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_red, 2)

        cv2.putText(frame, f"ID: {track_id}", (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_red, 2)
        cv2.putText(frame, f"Score: {cur_score:.2f}", (x1 + 10, y1 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_red, 2)

    model_status = "MODEL ON" if detection_type else "MODEL OFF"
    model_color = (0, 255, 0) if detection_type else (0, 0, 255)
    cv2.putText(frame, model_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, model_color, 2)

    return frame

def save_snapshot(timestamp_ms: int = None) -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        raise ValueError("Ошибка при открытии видеофайла")
    print('Видео файл обрабатывается, ожидайте ...')
    frame_number = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Не удалось прочитать следующий кадр")
            break
        frame = resize_image_proportionally(frame)

        frame_number += 1
        
        current_msec = int(cap.get(cv2.CAP_PROP_POS_MSEC))
        if current_msec >= timestamp_ms:
            print(f'Для того чтобы добрать до фрагмента {current_msec} мс нужно пропустить {frame_number} кадров.')
            snapshot_name = f"snapshot_{current_msec}.jpg"
            cv2.imwrite(snapshot_name, frame)
            print(f'Файл {snapshot_name} сохранён.')
            break



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Сохранение кадра из видео по времени (в мс).")
    parser.add_argument("timestamp_ms", type=int, help="Время в миллисекундах, на котором сохранить кадр.")

    args = parser.parse_args()
    save_snapshot(timestamp_ms=args.timestamp_ms)