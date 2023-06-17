import argparse
import helper_main

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
                    default=(400, 400), help="height and width dimensions of frame")
    args = vars(ap.parse_args())

    helper_main.main_init(args)
    helper_main.main()
