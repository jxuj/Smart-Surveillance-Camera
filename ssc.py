import collections
import cv2
import face_recognition.detect_face as detect_face
import face_recognition.facenet as facenet
import math
import numpy as np
import os
import pickle
import sys
import tensorflow as tf
import time
import urllib.request as ur

from datetime import datetime
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from timeit import default_timer as timer
from sklearn.svm import SVC 

# path to the user setting file
SETTING_PATH = 'setting.txt'
# read user settings from the setting text file
setting_file = open(SETTING_PATH, 'r')
# define the IP webcam to be used as the input
setting_file.readline()
URL = str(setting_file.readline())

# program starting time
start_time = datetime.now()
# path to the object detection log file, making sure there's no invalid characters in the file name
OBJECT_DETECTION_LOG_PATH = 'object_detection/object-' + str(start_time.date()) + '-' + str(start_time.time()).replace(':', '-') + '.txt'
# path to the face recognition log file, making sure there's no invalid characters in the file name
FACE_RECOGNITION_LOG_PATH = 'face_recognition/face-' + str(start_time.date()) + '-' + str(start_time.time()).replace(':', '-') + '.txt'

# variables for calculating fps
fps_count_started = False
fps_count_start_time = 0.0
fps_count_end_time = 0.0
fps_count_num_of_frames = 0

"""
Object detection.
Variables.
"""
object_detection_initialised = False
object_detection_on = False

# path to the user setting file for object detection
OBJECT_DETECTION_SETTING_PATH = 'object_detection/object_detection_setting.txt'

# path to object detection models
OBJECT_DETECTION_MODEL_PATH = 'models/object_detection/'

# user setting
# read user settings from the setting text file
object_detection_setting_file = open(OBJECT_DETECTION_SETTING_PATH, 'r')

# define the object detection model to be used
object_detection_setting_file.readline()
object_detection_model_name = object_detection_setting_file.readline()
# get rid of the line break at the end of the line just read
object_detection_model_name = object_detection_model_name.rstrip('\n')

# path to the frozen detection graph, which is the actual model used to perform object detection
OBJECT_DETECTION_CKPT_PATH = OBJECT_DETECTION_MODEL_PATH + object_detection_model_name + '/frozen_inference_graph.pb'
# path to the label map consisting of labels to be added to corresponding detection boxes
OBJECT_DETECTION_LABELS_PATH = OBJECT_DETECTION_MODEL_PATH + object_detection_model_name + '/oid_v5_label_map_customised.pbtxt'

# define the max number of classes of objects to be detected
object_detection_setting_file.readline()
max_num_classes_object = int(object_detection_setting_file.readline())

# define which classes of objects to be detected
selected_classes_object = []
object_detection_setting_file.readline()
for i in range(max_num_classes_object):
	object_detection_setting_file.readline()
	class_setting = int(object_detection_setting_file.readline())
	if class_setting == 1:
		selected_classes_object.append(i+1)

label_map_object = label_map_util.load_labelmap(OBJECT_DETECTION_LABELS_PATH)
categories_object = label_map_util.convert_label_map_to_categories(label_map_object, max_num_classes=max_num_classes_object, use_display_name=True)
category_index_object = label_map_util.create_category_index(categories_object)

# load the object detection model into memory
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(OBJECT_DETECTION_CKPT_PATH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
sess_object = tf.Session(graph=detection_graph)
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

"""
Face recognition.
Variables.
"""
face_recognition_on = False

# path to the user setting file for face recognition
FACE_RECOGNITION_SETTING_PATH = 'face_recognition/face_recognition_setting.txt'

# path to face recognition models.
FACE_RECOGNITION_MODEL_PATH = 'models/face_recognition/'

# path to the model used to perform face detection.
FACE_RECOGNITION_CKPT_PATH = FACE_RECOGNITION_MODEL_PATH + '20180402-114759.pb'

# path to the model used to perform face recognition.
FACE_RECOGNITION_CLASSIFIER_PATH = FACE_RECOGNITION_MODEL_PATH + 'my_classifier.pkl'

# path to the label map consisting of labels to be added to corresponding detection boxes.
FACE_RECOGNITION_LABELS_PATH = FACE_RECOGNITION_MODEL_PATH + 'facenet_label_map.pbtxt'

# user setting
# read user settings from the setting text file
face_recognition_setting_file = open(FACE_RECOGNITION_SETTING_PATH, 'r')

# define the max number of classes of faces to be detected
face_recognition_setting_file.readline()
max_num_classes_face = int(face_recognition_setting_file.readline())

# define the size of the input to be resized to
face_recognition_setting_file.readline()
input_image_size_face = int(face_recognition_setting_file.readline())

# define the minimum face size to be detected
face_recognition_setting_file.readline()
minsize_face = int(face_recognition_setting_file.readline())

# define the three steps face detection threshold
threshold_detection_face = [0.0, 0.0, 0.0]
face_recognition_setting_file.readline()
for i in range(3):
    threshold_detection_face[i] = float(face_recognition_setting_file.readline())

# define the factor used to create a scaling pyramid of face sizes to detect in the image
face_recognition_setting_file.readline()
factor_face = float(face_recognition_setting_file.readline())

# define the face recognition threshold
face_recognition_setting_file.readline()
threshold_recognition_face = float(face_recognition_setting_file.readline())

label_map_face = label_map_util.load_labelmap(FACE_RECOGNITION_LABELS_PATH)
categories_face = label_map_util.convert_label_map_to_categories(label_map_face, max_num_classes=max_num_classes_face, use_display_name=True)
category_index_face = label_map_util.create_category_index(categories_face)

# load The Custom Classifier
with open(FACE_RECOGNITION_CLASSIFIER_PATH, 'rb') as file:
    model, class_names = pickle.load(file)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess_face = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

# load the model
facenet.load_model(FACE_RECOGNITION_CKPT_PATH)

# get input and output tensors
images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
embedding_size = embeddings.get_shape()[1]

pnet, rnet, onet = detect_face.create_mtcnn(sess_face, "./face_recognition")

while(True):
    image = ur.urlopen(URL)
    image_array = np.array(bytearray(image.read()),dtype=np.uint8)
    frame = cv2.imdecode(image_array,-1)

    # dimension of the input image
    frame_shape = frame.shape

    """
    Object detection.
    Runtime.
    """
    if object_detection_initialised == False or object_detection_on == True:
        frame_expanded = np.expand_dims(frame, axis=0)
        (boxes_object, scores_object, classes_object, num_object) = sess_object.run(
            [detection_boxes, detection_scores, detection_classes, num_detections],
            feed_dict={image_tensor: frame_expanded})

        if object_detection_initialised == True:
            num_of_objects_detected = int(num_object[0])
            for i in range(0, num_of_objects_detected):
                # only draw objects of selected classes
                if int(classes_object[0][i]) not in selected_classes_object:
                    boxes_object[0][i] = [float(0), float(0), float(0), float(0)]
                    scores_object[0][i] = float(0)
                    classes_object[0][i] = float(1)
                    num_object[0] = num_object[0] - 1
                # report objects of selected classes once detected
                else:
                    with open(OBJECT_DETECTION_LOG_PATH, 'a') as log_file:
                        log_file.write('Time: ' + str(datetime.now()) + '\tCategory: ' + str(int(classes_object[0][i])) + '\tScore: ' + str(scores_object[0][i]) + '\n')

            # visualise the detection results.
            vis_util.visualize_boxes_and_labels_on_image_array(
                frame,
                np.squeeze(boxes_object),
                np.squeeze(classes_object).astype(np.int32),
                np.squeeze(scores_object),
                category_index_object,
                use_normalized_coordinates=True,
                line_thickness=8,
                min_score_thresh=0.60)

        # only run initialisation once.
        if object_detection_initialised == False:
            object_detection_initialised = True

    """
    Face recognition.
    Runtime.
    """
    if face_recognition_on == True:
        bounding_boxes_face, _ = detect_face.detect_face(frame, minsize_face, pnet, rnet, onet, threshold_detection_face, factor_face)
                
        faces_found = bounding_boxes_face.shape[0]

        boxes_face = [[[float(0),float(0),float(0),float(0)]] * (faces_found+1)]
        scores_face = [[float(0)] * (faces_found+1)]
        classes_face = [[float(0)] * (faces_found+1)]

        try:
            if faces_found > 0:
                det_face = bounding_boxes_face[:, 0:4]
                bb_face = np.zeros((faces_found, 4), dtype=np.int32)

                for i in range(faces_found):
                    bb_face[i][0] = det_face[i][0]
                    bb_face[i][1] = det_face[i][1]
                    bb_face[i][2] = det_face[i][2]
                    bb_face[i][3] = det_face[i][3]
                
                    cropped_face = frame[bb_face[i][1]:bb_face[i][3], bb_face[i][0]:bb_face[i][2], :]
                    scaled_face = cv2.resize(cropped_face, (input_image_size_face, input_image_size_face), interpolation=cv2.INTER_CUBIC)
                    scaled_face = facenet.prewhiten(scaled_face)
                    reshaped_face = scaled_face.reshape(-1, input_image_size_face, input_image_size_face, 3)
                    embed_array_face = sess_face.run(embeddings, feed_dict={images_placeholder: reshaped_face, phase_train_placeholder: False})
                    predictions_face = model.predict_proba(embed_array_face)
                    best_class_indices_face = np.argmax(predictions_face, axis=1)
                    best_class_score_face = predictions_face[np.arange(len(best_class_indices_face)), best_class_indices_face]
                    best_name_face = class_names[best_class_indices_face[0]]

                    # get relative coordinates of detection boxes
                    boxes_face[0][i] = [float(bb_face[i][1])/frame_shape[0], float(bb_face[i][0])/frame_shape[1], float(bb_face[i][3])/frame_shape[0], float(bb_face[i][2])/frame_shape[1]]
                    # the confidence score of a face is the one of its best match
                    scores_face[0][i] = float(best_class_score_face)

                    # a face is considered being recognised as someone when the best match has a score higher than the threshold
                    if best_class_score_face > threshold_recognition_face:
                        classes_face[0][i] = float(best_class_indices_face[0] + 2)
                    # otherwise the face detected is considered unknown
                    else:
                        classes_face[0][i] = float(1)
                        # report unknown faces once detected
                        with open(FACE_RECOGNITION_LOG_PATH, 'a') as log_file:
                            log_file.write('Time: ' + str(datetime.now()) + '\tScore: ' + str(scores_face[0][i]) + '\n')
                    
                # visualise the detection and recognition results.
                vis_util.visualize_boxes_and_labels_on_image_array(
                    frame,
                    np.squeeze(boxes_face),
                    np.squeeze(classes_face).astype(np.int32),
                    np.squeeze(scores_face),
                    category_index_face,
                    use_normalized_coordinates=True,
                    line_thickness=8)
        except:
            pass

    # display the result image
    cv2.imshow('Smart Surveillance Camera', frame)

    # increment number of frames being processed by one for calculating FPS
    fps_count_num_of_frames = fps_count_num_of_frames + 1

    # handle user input
    key = cv2.waitKey(1)
    # press 'q' to exit
    if key == ord('q'):
        break
    # press 'o' to switch object detection on and off
    elif key == ord('o'):
        object_detection_on = not object_detection_on
    # press 'f' to switch face recognition on and off
    elif key == ord('f'):
        face_recognition_on = not face_recognition_on
    # press 'p' to switch fps calculation on and off
    elif key == ord('p'):
        # initialise and start the fps calculation if it's not already started
        if fps_count_started == False:
            fps_count_num_of_frames = 0
            fps_count_start_time = timer()
            fps_count_started = True
        # stop, calculate and display the fps if it's already started
        else:
            fps_count_started = False
            fps_count_end_time = timer()
            fps = fps_count_num_of_frames / (fps_count_end_time - fps_count_start_time)
            print('FPS:' + str(fps))

cv2.destroyAllWindows()