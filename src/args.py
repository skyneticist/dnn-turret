import argparse


ap = argparse.ArgumentParser()
ap.add_argument("-i", "--ip", type=str, required=False, default="127.0.0.1",
                help="ip address of device")
ap.add_argument("-o", "--port", type=int, required=False, default="8080",
                help="ephemeral port number of server (1024 to 65535)")
ap.add_argument("-f", "--frame-count", type=int, default=32,
                required=False, help="# of frames used to construct background model")
ap.add_argument("-w", "--write", type=bool, required=False,
                default=False, help="write to microcontroller")
ap.add_argument("-p", "--prototxt", required=False,
                default="src/dnn/deploy.prototxt", help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False,
                default="src/dnn/300_300.caffemodel", help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter out weak detections")
ap.add_argument("-r", "--resize", type=tuple, required=False,
                default=(900, 600), help="height and width dimensions of frame")
_args = vars(ap.parse_args())

args = {
    "ip": _args["ip"],
    "port": _args["port"],
    "write": _args["write"],
    "prototxt": _args["prototxt"],
    "model": _args["model"],
    "confidence": _args["confidence"],
    "resize": _args["resize"]
}

