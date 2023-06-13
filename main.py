import numpy as np
import argparse
import time
import serial
import cv2

from VideoStream import VideoStream
from queue import LifoQueue
from threading import Thread


def serial_init():
    # establish connection with microcontroller via serial (USB)
    print("connecting to uController...")
    uContoller = serial.Serial(port='COM4', baudrate=115200, timeout=.1)
    time.sleep(2.0)
    print("connected!")


# only write when face_detecting is True
def write_cv_data(q) -> None:
    while True:
        if face_detecting:
            # float ?
            data: float = q.get()
            # uController.write(bytes(data, 'utf-8'))
            time.sleep(0.01)  # need to test for best value
        else:
            # clear the LIFO queue if no longer
            # detecting a face - ensure mutex lock
            with q.mutex:
                q.queue.clear()


def resize_img(img, height, width):
    dim = (width, height)
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)


def main():
    print("starting threaded video stream...")
    vs: VideoStream = VideoStream().start()

    print("loading dnn caffe model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    print("creating named window and resizing it...")
    cv2.namedWindow('processed_img', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('processed_img', args["resize"][0], args["resize"][1])

    print("dnn-turret is actively scanning...")

    # last in first out queue is essential for
    # efficient parity between threaded serial task
    # and main-threaded cv draw computations/updates
    q: LifoQueue = LifoQueue()

    # fps
    new_frame_time = 0
    last_frame_time = 0

    # data check
    prevData: str = ""

    dynamic_centroid_diameter = 1

    # create a separate thread to handle
    # sending serial updates via write_cv_data method
    if args["write"]:
        serial_init()
        th_ucontroller: Thread = Thread(target=write_cv_data, args=(q,))
        th_ucontroller.daemon = True
        th_ucontroller.start()

    while 1:
        img = vs.read()
        # img = resize_img(img, 800, 600)

        new_frame_time = time.time()
        fps = int(1 / (new_frame_time - last_frame_time))
        last_frame_time = new_frame_time

        # putting the FPS count on the frame
        cv2.putText(img, str(fps), (7, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    3, (100, 255, 0), 3, cv2.LINE_AA)

        # get the center of the frame/img
        (h, w) = img.shape[:2]
        frame_center_xy = (w//2, h//2)

        cv2.circle(
            img, (frame_center_xy[0], frame_center_xy[1]), dynamic_centroid_diameter, (255, 200, 1), -1)

        blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            face_detecting = True
            confidence = detections[0, 0, i, 2]

            if confidence < args["confidence"]:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            box_width = endX - startX
            box_height = endY - startY

            bboxes = np.shape((30, 1))
            np.append(bboxes, box_width + box_height)
            closest_bbox = np.max(bboxes)

            bbox_color = (0, 50, 200) if closest_bbox is box_width + \
                box_height else (200, 25, 10)

            box_center_x = int((endX + startX)/2)
            box_center_y = int((endY + startY)/2)

            # serial data sent to arduino
            data: str = "X{0:d}Y{1:d}Z".format(box_center_x, box_center_y)

            # if data is different, put data in LIFO queue
            if data != prevData:
                q.put(data)
                prevData = data

            confidence_text = "{:2f}%".format(confidence * 100)
            x = startX - 10 if startX - 10 > 10 else startX + 10
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(img, (startX, startY), (endX, endY), bbox_color, 2)
            cv2.putText(img, confidence_text, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (44, 0, 100), 2)

            dlt = abs(
                (frame_center_xy[0] + frame_center_xy[1]) - (box_center_x + box_center_y))

            dynamic_centroid_diameter = int((dlt / 100) + 3)

            # line denoting distance between center of
            # camera view (img) and the center of detected faces
            cv2.line(img, (frame_center_xy[0], frame_center_xy[1]), (
                box_center_x, box_center_y), (250 - (dlt * 0.3), 200 - (dlt * 0.3), 245), round((dlt / 100) + 1)
            )

            # draw center bounding box coords on screen
            cv2.putText(img, ("X:" + str(box_center_x) + ", Y:" + str(box_center_y)),
                        (y, x), cv2.FONT_HERSHEY_COMPLEX, 0.8, (144, 125, 10), 1, cv2.LINE_AA)

        cv2.imshow("output", img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    vs.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--write", type=bool, required=False,
                    default=False, help="write to microcontroller")
    ap.add_argument("-p", "--prototxt", required=False,
                    default="dnn/deploy.prototxt", help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=False,
                    default="dnn/300_300.caffemodel", help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
                    help="minimum probability to filter out weak detections")
    ap.add_argument("-r", "--resize", type=tuple, required=False,
                    default=(400, 300), help="height and width dimensions of frame")
    args = vars(ap.parse_args())

    # global
    # write to arduino only when detecting
    face_detecting: bool = False

    main()
