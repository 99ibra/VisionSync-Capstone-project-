import cv2
import numpy as np
import time
import pyttsx3  # text to speech library

engine = pyttsx3.init()  # initialize text to speech
engine.setProperty('rate', 150)
engine.setProperty('volume', 1)

def warn_obstacle(side):
    engine.say(f"Warning: {side}")
    engine.runAndWait()


# Function to detect objects using SSD MobileNet V3
def detect_objects(frame, model, classes):
    classIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)
    boxes = []

    if len(classIndex) > 0:
        for classInd, conf, box in zip(classIndex.flatten(), confidence.flatten(), bbox):
            if classInd == 1:  # Class 1 corresponds to 'person' in COCO dataset
                x1, y1, w, h = box
                boxes.append([int(x1), int(y1), int(w), int(h)])  # Append to list
    
    return boxes


# Function to match objects between left and right images
def match_objects(left_boxes, right_boxes):
    matched_objects = []

    for left_box in left_boxes:
        min_distance = float('inf')
        matched_right_box = None

        for right_box in right_boxes:
            # Calculate distance between bounding box centers
            distance = abs((left_box[0] + left_box[2] / 2) - (right_box[0] + right_box[2] / 2))

            if distance < min_distance:
                min_distance = distance
                matched_right_box = right_box

        if matched_right_box is not None:
            matched_objects.append((left_box, matched_right_box))
    
    return matched_objects


# Depth equation
def calculate_depth(disparity, baseline, focal_length):
    if disparity == 0:
        return float('inf')  # avoid division by zero which gives error
    else:
        depth = (baseline * focal_length) / disparity
        return depth


# These variables are for the text-to-speech
warning_delay = 7
last_warning_time = 0


# Visualize objects with depth through one of the rectified video feeds
def visualize_objects_with_depth(frame, boxes, depths):
    height, width, _ = frame.shape
    middle_point = width // 2  # Middle point of the video feed

    global last_warning_time

    print("Number of detected objects:", len(boxes))
    print("Number of calculated depths:", len(depths))
    for i, box in enumerate(boxes):
        x, y, w, h = box
        depth = depths[i] if i < len(depths) else None

        # Bounding box center
        center_x = x + w // 2

        # Determine if the object is on the left or right side
        if center_x < middle_point:
            side_label = "Left"
        else:
            side_label = "Right"

        # Label the bounding box with depth and side
        if depth is not None:
            if depth < 1.5:
                color = (0, 0, 255)  # Red when close
                warning_label = "Obstacle Close"
                current_time = time.time()  # Get the current time
                # Check if enough time has passed since the last warning
                if current_time - last_warning_time >= warning_delay:
                    # Warn about obstacle
                    warn_obstacle(side_label)
                    last_warning_time = current_time  # Update last warning time
            
            else:
                color = (0, 255, 0)  # Green when far
                warning_label = ""
            
            cv2.putText(frame, f'Depth: {depth:.2f} meters ({side_label})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if warning_label:
                cv2.putText(frame, warning_label, (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        else:
            cv2.putText(frame, f'({side_label})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Line in the middle to differentiate between left and right objects
        cv2.line(frame, (middle_point, 0), (middle_point, height), (255, 0, 0), 2)


# Paths to the model files for SSD MobileNet V3
config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'frozen_inference_graph.pb'

# Load the model
model = cv2.dnn_DetectionModel(frozen_model, config_file)

# Load class labels
classLabels = []
filename = 'labels.txt'
with open(filename, 'rt') as spt:
    classLabels = spt.read().rstrip('\n').split('\n')

# Set model parameters
model.setInputSize(320, 320)
model.setInputScale(1.0 / 127.5)
model.setInputMean((127.5, 127.5, 127.5))
model.setInputSwapRB(True)

# Load stereo rectification maps
cv_file = cv2.FileStorage('stereoMap.xml', cv2.FileStorage_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# Open stereo video feeds
left_video_feed = cv2.VideoCapture(1, cv2.CAP_DSHOW)
right_video_feed = cv2.VideoCapture(0, cv2.CAP_DSHOW)

while left_video_feed.isOpened() and right_video_feed.isOpened():
    success_right, frame_right = right_video_feed.read()
    success_left, frame_left = left_video_feed.read()

    if not (success_left and success_right):
        break

    # Undistort and rectify images
    frame_left_rectified = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_right_rectified = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
    # Detect objects using SSD MobileNet V3
    left_boxes = detect_objects(frame_left_rectified, model, classLabels)
    right_boxes = detect_objects(frame_right_rectified, model, classLabels)

    # Match objects between stereo images
    matched_objects = match_objects(left_boxes, right_boxes)

    # Iterate over matched objects and estimate depth
    depths = []
    for left_obj, right_obj in matched_objects:
        # Calculate pixel disparity between left and right bounding boxes
        disparity = abs(left_obj[0] - right_obj[0])

        baseline = 72.0  # Adjust this value based on your stereo camera setup
        focal_length = 4.0  # Adjust this value based on your camera's focal length

        # Estimate depth
        depth = calculate_depth(disparity, baseline, focal_length)
        depths.append(depth)

    # Visualize objects with depth in the rectified video feed
    visualize_objects_with_depth(frame_left_rectified, left_boxes, depths)

    # Display the video feed with labels and depth estimated
    cv2.imshow('Rectified Video Feed with Depth', frame_left_rectified)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video feeds and close windows
left_video_feed.release()
right_video_feed.release()
cv2.destroyAllWindows()
