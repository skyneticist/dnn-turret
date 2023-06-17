import numpy as np
import time
import cv2

import config

from VideoStream import VideoStream
from queue import LifoQueue
from threading import Thread

from helper import compute_aggression, write_cv_data, resize_img


def main_init(args):
    global vs
    global net
    global detection_confidence
    global q

    print("starting threaded video stream...")
    vs = VideoStream().start()

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


def main():
    # fps
    new_frame_time = 0
    last_frame_time = 0

    # data check
    previous_data: str = ""

    # init center circle size
    dynamic_centroid_diameter = 1

    bbb = []

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

        bbb.clear()

        for i in range(0, detections.shape[2]):
            config.face_detecting = True

            confidence = detections[0, 0, i, 2]

            if confidence < detection_confidence:
                continue

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            box_width = endX - startX
            box_height = endY - startY
            average_box_size = (box_width + box_height) / 2

            compute_aggression(img, average_box_size / 500.0)

            bbb.append(box)
            bboxes = np.array(bbb)
            # print("bboxes: {}".format(bboxes))
            np.append(bboxes, average_box_size)

            closest_bbox = np.max(bboxes)
            print("Closest bBox: {}".format(closest_bbox))

            bbox_color = (10, 15, 225)
            print(np.where(bboxes == closest_bbox))

            bbox_color = (255, 255 - (average_box_size / 10) + 1, 255)

            box_center_x = int((endX + startX)/2)
            box_center_y = int((endY + startY)/2)

            # serial data sent to arduino
            data: str = "X{0:d}Y{1:d}Z".format(box_center_x, box_center_y)
            # print("data: {}".format(data))

            # if data is different, put data in LIFO queue
            if data != previous_data:
                q.put(data)
                previous_data = data

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
