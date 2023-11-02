import threading
from flask import Response
from flask import Flask
from flask import render_template
from src.main_init import convert_frame

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(convert_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")


def flask_thread_start(args):
    t = threading.Thread(target=lambda: app.run(
        host=args["ip"], port=args["port"], debug=True, use_reloader=False))
    t.daemon = True
    t.start()
