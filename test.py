import cv2
from config import VIDEO_PATH
from utils import resize_image_proportionally


def frame_to_timestamp() -> dict:
    result = {}

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

        frame_number += 1
        
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        result[frame_number] = current_time
    cap.release()
    return result


def find_best_match(frame_to_timestamp: dict, target_timestamp_ms: float = None):
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    if not cap.isOpened():
        raise ValueError("Ошибка при открытии видеофайла")
    print('Видео файл обрабатывается, ожидайте ...')

    best_timestamp = None
    best_frame_number = None
    best_diff = float("inf")

    for current_frame_number, current_timestamp in frame_to_timestamp.items():
        diff = abs(current_timestamp - target_timestamp_ms)
        if diff < best_diff:
            best_diff = diff
            best_timestamp = current_timestamp
            best_frame_number = current_frame_number
        elif current_timestamp > target_timestamp_ms:
            break

    if best_timestamp is not None:
        cap.set(cv2.CAP_PROP_POS_MSEC, best_timestamp)
        success, frame = cap.read()
        frame = resize_image_proportionally(frame)
        print(f'Для того чтобы добрать до фрагмента {target_timestamp_ms} мс нужно пропустить {best_frame_number} кадров.')
        snapshot_name = f"res_1_snapshot_{target_timestamp_ms}.jpg"
        cv2.imwrite(snapshot_name, frame)
        print(f'Файл {snapshot_name} сохранён.')
    cap.release()

def save_snapshot(target_timestamp_ms: float = None) -> None:
    cap = cv2.VideoCapture(VIDEO_PATH)
    best_frame = None
    best_frame_number = None

    best_diff = float("inf")
    
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
        
        current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
        diff = abs(current_time - target_timestamp_ms)
        if diff < best_diff:
            best_diff = diff
            best_frame = frame
            best_frame_number = frame_number

        elif current_time > target_timestamp_ms:
            break


    if best_frame is not None:
        print(f'Для того чтобы добрать до фрагмента {target_timestamp_ms} мс нужно пропустить {best_frame_number} кадров.')
        snapshot_name = f"res_2_snapshot_{target_timestamp_ms}.jpg"
        cv2.imwrite(snapshot_name, best_frame)
        print(f'Файл {snapshot_name} сохранён.')
    cap.release()
        

target_timestamp_ms = 15000
# можно сохранить все временные метки кадров, и потом по ним "перемещаться"
result = frame_to_timestamp()
find_best_match(frame_to_timestamp=result, target_timestamp_ms=target_timestamp_ms)

# либо можно искать "ближайшую" временную метку в процессе считывания видео
save_snapshot(target_timestamp_ms=target_timestamp_ms)
