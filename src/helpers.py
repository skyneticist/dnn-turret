import time
import serial
from src.args import args
import cv2


def resize_img(img, height, width) -> any:
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def compute_aggression(img, al):
    fill_color = (5, 145, 66)
    fill = int(img.shape[0] * al)
    cv2.rectangle(img, (0, img.shape[1] - fill),
                  (20, img.shape[0]), fill_color, -1)


def serial_init() -> serial:
    if args["write"]:
        # establish connection with microcontroller via serial
        print("connecting to uController...")
        uController = ser = serial.Serial(
            port="/dev/cu.usbmodem1101",
            baudrate=115200,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=1,
            xonxoff=False,
            rtscts=False,
            dsrdtr=False,
            writeTimeout=2
        )
        time.sleep(1.0)
        print("connected!")
        return uController
    return None
