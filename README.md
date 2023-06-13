# dnn-turret
Deep Neural Network OpenCV Face Tracking test for auto turret

Simple to run:


```Bash
git clone https://github.com/skyneticist/dnn-turret/
cd dnn-turret
python dl_detect.py
```

This will run face detection using Open CV and DNN (Deep Neural Network) while drawing a line denoting distance between center of image and center of any bounding box of a detected face.

Line color and width is dynamic and updates in real-time based on the length of the line.
