import time
import serial
import cv2
import jido.config as config


def resize_img(img, height, width) -> any:
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def compute_aggression(img, al):
    fill_color = (5, 145, 66)
    fill = int(img.shape[0] * al)
    cv2.rectangle(img, (0, img.shape[1] - fill),
                  (20, img.shape[0]), fill_color, -1)


def serial_init() -> serial:
    # establish connection with microcontroller via serial
    print("connecting to uController...")
    uController = serial.Serial(baudrate=115200, timeout=.1)
    time.sleep(2.0)
    print("connected!")
    return uController


# only write when face_detecting is True
def write_cv_data(q) -> None:
    while True:
        if config.face_detecting:
            # float ?
            data: float = q.get()
            config.uController.write(bytes(data, 'utf-8'))
            time.sleep(0.01)  # need to test for best value
        else:
            # clear the LIFO queue if no longer
            # detecting a face - ensure mutex lock
            with q.mutex:
                q.queue.clear()
