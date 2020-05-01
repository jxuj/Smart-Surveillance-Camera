# Smart Surveillance Camera

This repository contains a Python program capable of running object detection and face recognition on a remote camera with models trained using TensorFlow which can be potentially used for building intelligent video surveillance systems.

For object detection, a list of pre-trained models can be found at [Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md).

For face recognition, this project works with models trained with [Face Recognition using Tensorflow](https://github.com/davidsandberg/facenet).

## Compatibility

This project is tested using TensorFlow 1.14.0, Python 3.6, scipy 1.1.0 and numpy 1.16.2.

## Setup

1. Assuming that Python 3.6 is installed, install dependencies from the root directory.
```Shell
pip install -r requirements.txt
```

2. Place trained models in corresponding sub-directories under the models directory which can be found under the root directory. Pre-trained models for face detection can be found at [Face Recognition using Tensorflow](https://github.com/davidsandberg/facenet), which should also be placed under face_recognition.

3. Modify label map files which can be found in sub-directories under the models directory. Note that label maps for object detection models should be placed under the specific model directory with one file per model.

4. Modify setting files which can be found in object_detection/face_recognition under the root directory.
For example, the name of the directory containing the object detection model to be used has to be defined in the object detection setting file.

## Run the program

After successfully completing the setup steps, you'll be ready to run the program except that you have to define the URL of the remote camera used as inputs to this program in the setting file under the root directory.

To run the program
```Shell
python ssc.py
```

## User inputs

After the program is just started, both object detection and face recognition are switched off by default.

Press 'o' to switch object detection on and off. Press 'f' to switch face recognition on and off.

To get the average FPS of the program for a certain amount of time, press 'p', wait a while, and press 'p' again.
The average period of that period of time will be reported at the command prompt.

You can quit the program by pressing 'q'.

## Logging

After the program is started, a log file will be created for object detection when any object is detected,
which will be used to record information of all successful detections before quitting the program.

A log file will also be created for face recognition when an unknown face is spotted for the first time,
and is only going to record information of unknown faces detected and recognised.

All log files are named after the starting time of the program.
Therefore, every time the program is started, at most two log files will be created for this run.
New log files will be created for new instance of the program.
