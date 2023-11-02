from src.args import args
import cv2
import numpy as np
import threading
import time

from src.helpers import compute_aggression, resize_img, serial_init
from threading import Thread
from queue import LifoQueue
from src.VideoStream import VideoStream


def main_init():
    global vs
    global net
    global detection_confidence
    global q

    print("starting threaded video stream...")
    vs = VideoStream(1).start()

    print("loading dnn caffe model...")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

    detection_confidence = args["confidence"]
    print("detection confidence set to {}".format(detection_confidence))

    print("creating named window and resizing it...")
    cv2.namedWindow('output', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('output', args["resize"][0], args["resize"][1])

    # last in first out queue is essential for
    # efficient parity between threaded serial task
    # and main-threaded cv draw computations/updates
    q = LifoQueue()

    if args["write"]:
        print("starting thread for writing data...")
        th_ucontroller: Thread = Thread(target=write_cv_data, args=(q,))
        th_ucontroller.daemon = True
        th_ucontroller.start()

    print("✅: system startup completed successfully\n")
    print("📠 dnn-turret is actively scanning...")

    return args


def convert_frame():
    while True:
        with lock:
            if outputFrame is None:
                continue
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            if not flag:
                continue

        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + bytearray(encodedImage) + b'\r\n')


# only write when face_detecting is True
def write_cv_data(q, uController, face_detecting) -> None:
    while True:
        if face_detecting:
            # float ?
            data: float = q.get()
            uController.write(bytes(data, 'utf-8'))
            time.sleep(0.01)  # need to test for best value
        else:
            # clear the LIFO queue if no longer
            # detecting a face - ensure mutex lock
            with q.mutex:
                q.queue.clear()


def video_writer_init(framerate):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('out.mp4', fourcc, framerate,
                          (args["resize"][0], args["resize"][1]))
    return out


def load_classnames():
    class_names = []
    # load the COCO class names
    with open('src/dnn/object_detection_classes_coco.txt', 'r') as f:
        class_names = f.read().split('\n')
    return class_names


def device_write_thread(uController):
    thread = threading.Thread(target=write_cv_data, args=(
        [q, uController, face_detecting]))
    thread.daemon = True
    return thread


def main():
    global face_detecting, lock, outputFrame

    face_detecting = True

    outputFrame = None
    lock = threading.Lock()

    # fps
    new_frame_time = 0
    last_frame_time = 0

    # data check
    previous_data: str = ""

    bbb = []

    uController = serial_init()
    if args["write"]:
        device_write_thread(uController).start()

    dynamic_centroid_diameter = 1
    class_names = load_classnames()

    # get a different color array for each of the classes
    CLASSNAME_COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

    write_framerate = 30.0
    out = video_writer_init(write_framerate)

    while 1:
        img = vs.read()
        img = resize_img(img, 600, 900)

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

        bbb.clear()

        image_height, image_width, _ = img.shape

        # for detection in range(0, detections.shape[2]):
        for detection in detections[0, 0, :, :]:
            face_detecting = True

            confidence = detection[2]
            # confidence = detections[0, 0, detection, 2]

            if confidence < detection_confidence:
                continue

            class_id = detection[1]
            class_name = class_names[int(class_id)-1]
            color = CLASSNAME_COLORS[int(class_id)]

            # box = detections[0, 0, detection, 3:7] * np.array([w, h, w, h])
            # get the bounding box coordinates
            box_x = detection[3] * image_width
            box_y = detection[4] * image_height
            # get the bounding box width and height
            box_width = detection[5] * image_width
            box_height = detection[6] * image_height

            # (startX, startY, endX, endY) = box.astype("int")

            # box_width = endX - startX
            # box_height = endY - startY
            average_box_size = (box_width + box_height) / 2

            compute_aggression(img, average_box_size / 500.0)

            # bbb.append(box)
            bboxes = np.array(bbb)
            np.append(bboxes, average_box_size)

            bbox_color = (255, 255 - (average_box_size / 10) + 1, 255)
            # closest_bbox = np.max(bboxes)
            # print("Closest bBox: {}".format(closest_bbox))

            box_center_x = int((box_x + box_width) // 2)
            box_center_y = int((box_y + box_height) // 2)
            # box_center_x = int((endX + startX)/2)
            # box_center_y = int((endY + startY)/2)

            # serial data sent to arduino
            data: str = "{0:d}~{1:d}".format(box_center_x, box_center_y)
            # print("data: {}".format(data))

            # if data is different, put data in LIFO queue
            if data != previous_data:
                q.put(data)
                previous_data = data

            cv2.rectangle(img, (int(box_x), int(box_y)), (int(
                box_width), int(box_height)), color, thickness=2)

            # confidence_text = "{:2f}%".format(confidence * 100)
            # x = startX - 10 if startX - 10 > 10 else startX + 10
            # y = startY - 10 if startY - 10 > 10 else startY + 10
            # cv2.rectangle(img, (startX, startY), (endX, endY), bbox_color, 2)

            # cv2.putText(img, confidence_text, (startX, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.65, (44, 0, 100), 2)

            dlt = abs(
                (frame_center_xy[0] + frame_center_xy[1]) - (box_center_x + box_center_y))

            dynamic_centroid_diameter = int((dlt / 100) + 3)

            # cv2.putText(img, str(dlt), (int(box_x), int(box_y - 5)),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # line denoting distance between center of
            # camera view (img) and the center of detected faces
            cv2.line(img, (frame_center_xy[0], frame_center_xy[1]), (
                box_center_x, box_center_y), (250 - (dlt * 0.3), 200 - (dlt * 0.3), 245), round((dlt / 100) + 1)
            )

            cv2.putText(img, class_name, (int(box_x), int(box_y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # draw center bounding box coords on screen
            cv2.putText(img, (str(box_center_x) + "-" + str(box_center_y)),
                        (int(box_x + 10), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (52, 152, 219), 2)

            with lock:
                outputFrame = img.copy()

        out.write(img)
        cv2.imshow('output', img)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    out.release()
    vs.stop()
