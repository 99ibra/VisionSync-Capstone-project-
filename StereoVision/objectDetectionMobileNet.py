import cv2
import numpy as np

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

# Function to detect objects using SSD MobileNet V3
def detect_objects(frame, model, classes):
    classIndex, confidence, bbox = model.detect(frame, confThreshold=0.5)
    boxes = []  # Initialize as a list

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

# Function to calculate depth from pixel disparity
def calculate_depth(disparity, baseline, focal_length):
    if disparity == 0:
        return float('inf')  # Return infinity if disparity is zero to avoid division by zero
    else:
        depth = (baseline * focal_length) / disparity
        return depth / 2

# Function to visualize objects with depth in the rectified video feed
def visualize_objects_with_depth(frame, boxes, depths):
    # print("Number of detected objects:", len(boxes))
    # print("Number of calculated depths:", len(depths))
    for i, box in enumerate(boxes):
        x, y, w, h = box
        if i < len(depths):
            depth = depths[i]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Depth: {depth:.2f} meters', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        else:
            print("Depth not available for object at index", i)

# Load stereo rectification maps
cv_file = cv2.FileStorage('stereoMap.xml', cv2.FileStorage_READ)
stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# Open stereo video feeds
left_video_feed = cv2.VideoCapture(1,cv2.CAP_DSHOW)
right_video_feed = cv2.VideoCapture(0,cv2.CAP_DSHOW)

while left_video_feed.isOpened() and right_video_feed.isOpened():
    succes_right, frame_right = right_video_feed.read()
    succes_left, frame_left = left_video_feed.read()

    if not (succes_left and succes_right):
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

        # Use triangulation to estimate depth
        depth = calculate_depth(disparity, baseline, focal_length)
        depths.append(depth)

    # Visualize objects with depth in the rectified video feed
    visualize_objects_with_depth(frame_left_rectified, left_boxes, depths)

    # Display rectified video feed with detected objects and estimated distances
    cv2.imshow('Rectified Video Feed with Depth', frame_left_rectified)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video feeds and close windows
left_video_feed.release()
right_video_feed.release()
cv2.destroyAllWindows()
