from jido.helpers import serial_init


def init(args):
    global uController
    global face_detecting
    uController = serial_init() if args["write"] else None
    face_detecting = False  # write to uController only when detecting
