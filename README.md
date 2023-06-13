# dnn-turret
Deep Neural Network OpenCV Face Tracking test for auto turret

- Uses Python Threading class to achieve ~115% gain in frames per second
- Uses a LIFO queue to pass data to microcontroller via Threaded function
- Work In Progress, but does run without any headache

Simple to run:


```Bash
git clone https://github.com/skyneticist/dnn-turret/
cd dnn-turret
pip install -r requirements.txt
python main.py
```

This will run face detection using Open CV and DNN (Deep Neural Network) while drawing a line denoting distance between center of image and center of any bounding box of a detected face.

Line color and width is dynamic and updates in real-time based on the length of the line.

Note: 

There are several flags and user-definable properties that can be set when calling the python `main.py` script.
These can be found near the bottom of the `main.py` file.

Example:

```
python main.py -w -r (1200, 900) -c 85 -p <path/to/different/prototxt_file> -m <path/to/other/caffe_model_file> 
```
