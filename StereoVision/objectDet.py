from ultralytics import YOLO
import cv2
import math 
from matplotlib import pyplot as plt
import numpy as np

# cv_file = cv2.FileStorage()
# cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

# stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
# stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
# stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
# stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

# left_video_feed =  cv2.VideoCapture(1, cv2.CAP_DSHOW)
# right_video_feed =  cv2.VideoCapture(0, cv2.CAP_DSHOW)

# while(left_video_feed.isOpened() and right_video_feed.isOpened()):

#     succes_right, frame_right = right_video_feed.read()
#     succes_left, frame_left = left_video_feed.read()

#     # # Undistort and rectify images
#     # frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
#     # frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    

    
#     # cv2.imshow("frame left", frame_left)
#     # cv2.imshow("frame right", frame_right) 



# Function to detect objects using YOLOv8
def detect_objects(frame, model, classes):
    results = model(frame)
    boxes = []
    for result in results.xyxy[0]:
        x1, y1, x2, y2, confidence, class_id = result
        if confidence > 0.5 and classes[int(class_id)] == "person":
            boxes.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])
    print(results)

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
    depth = (baseline * focal_length) / disparity
    return depth

# Function to visualize objects with depth in the rectified video feed
def visualize_objects_with_depth(frame, boxes, depths):
    for i, box in enumerate(boxes):
        x, y, w, h = box
        depth = depths[i]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, f'Depth: {depth:.2f} meters', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Load YOLOv8 model and classes
# net = cv2.dnn.readNet('yolov8.weights', 'yolov8.cfg')
# net = cv2.dnn.readNet('yolo-Weights/yolov8n.pt')
model = YOLO("yolo-Weights/yolov8n.pt")
classes = ["person"]

# # Load YOLOv8 model
# net = cv2.dnn.readNet('yolo-Weights/yolov8n.pt')

cv_file = cv2.FileStorage()
cv_file.open('stereoMap.xml', cv2.FileStorage_READ)

stereoMapL_x = cv_file.getNode('stereoMapL_x').mat()
stereoMapL_y = cv_file.getNode('stereoMapL_y').mat()
stereoMapR_x = cv_file.getNode('stereoMapR_x').mat()
stereoMapR_y = cv_file.getNode('stereoMapR_y').mat()

left_video_feed =  cv2.VideoCapture(1, cv2.CAP_DSHOW)
right_video_feed =  cv2.VideoCapture(0, cv2.CAP_DSHOW)

while(left_video_feed.isOpened() and right_video_feed.isOpened()):

    succes_right, frame_right = right_video_feed.read()
    succes_left, frame_left = left_video_feed.read()

    # Undistort and rectify images
    frame_left = cv2.remap(frame_left, stereoMapL_x, stereoMapL_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    frame_right = cv2.remap(frame_right, stereoMapR_x, stereoMapR_y, cv2.INTER_LANCZOS4, cv2.BORDER_CONSTANT, 0)
    
    # Detect objects using YOLOv8
    left_boxes = detect_objects(frame_left, model, classes)
    right_boxes = detect_objects(frame_right, model, classes)

    # Match objects between stereo images
    matched_objects = match_objects(left_boxes, right_boxes)

    # Iterate over matched objects and estimate depth
    depths = []
    for left_obj, right_obj in matched_objects:
        # Calculate pixel disparity between left and right bounding boxes
        disparity = abs(left_obj[0] - right_obj[0])

        baseline =72.0
        focal_length=4.0

        # Use triangulation to estimate depth (baseline and focal_length are assumed)
        depth = calculate_depth(disparity, baseline, focal_length)
        depths.append(depth)
    
    cv2.imshow("frame left", frame_left)
    cv2.imshow("frame right", frame_right)

    # Visualize objects with depth in the rectified video feed
    visualize_objects_with_depth(left_video_feed, left_boxes, depths)

    # Display rectified video feed with detected objects and estimated distances
    cv2.imshow('Rectified Video Feed with Depth', left_video_feed)

cv2.waitKey(0)
cv2.destroyAllWindows()
    
    # cv2.imshow("frame left", frame_left)
    # cv2.imshow("frame right", frame_right) 
    
    




# # # Load rectified video feeds from both cameras
# # left_video_feed = cv2.imread("frame left", frame_left)
# # right_video_feed = cv2.imread("frame right", frame_right)

# # Detect objects using YOLOv8
# left_boxes = detect_objects(left_video_feed, net, classes)
# right_boxes = detect_objects(right_video_feed, net, classes)

# # Match objects between stereo images
# matched_objects = match_objects(left_boxes, right_boxes)

# # Iterate over matched objects and estimate depth
# depths = []
# for left_obj, right_obj in matched_objects:
#     # Calculate pixel disparity between left and right bounding boxes
#     disparity = abs(left_obj[0] - right_obj[0])

#     baseline =72.0
#     focal_length=4.0

#     # Use triangulation to estimate depth (baseline and focal_length are assumed)
#     depth = calculate_depth(disparity, baseline, focal_length)
#     depths.append(depth)

# # Visualize objects with depth in the rectified video feed
# visualize_objects_with_depth(left_video_feed, left_boxes, depths)

# # Display rectified video feed with detected objects and estimated distances
# cv2.imshow('Rectified Video Feed with Depth', left_video_feed)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# ChatGPT suggestion on how to get the distance

# Load rectified video feeds from both cameras
# left_video_feed =  cv2.VideoCapture(1, cv2.CAP_DSHOW)
# right_video_feed =  cv2.VideoCapture(0, cv2.CAP_DSHOW)


# # Detect objects using YOLOv8
# left_objects = detect_objects(left_video_feed)
# right_objects = detect_objects(right_video_feed)

# # Match objects between stereo images
# matched_objects = match_objects(left_objects, right_objects)

# # Iterate over matched objects and estimate depth
# for left_obj, right_obj in matched_objects:
#     # Calculate pixel disparity between left and right bounding boxes
#     disparity = calculate_disparity(left_obj, right_obj)

#     # Use triangulation to estimate depth
#     depth = estimate_depth(disparity, baseline, focal_length)

#     # Visualize objects with depth in rectified video feed
#     visualize_objects_with_depth(left_video_feed, left_obj, depth)

# # Display rectified video feed with detected objects and estimated distances
# cv2.imshow('Rectified Video Feed with Depth', left_video_feed)
# cv2.waitKey(0)
# cv2.destroyAllWindows()





# # start webcam
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(3, 1280)
# cap.set(4, 720)

# # model
# model = YOLO("yolo-Weights/yolov8n.pt")

# # object classes
# classNames = ["person"]


# while True:
#     success, img = cap.read()
#     results = model(img, stream=True)

#     # coordinates
#     for r in results:
#         boxes = r.boxes

#         for box in boxes:
#             # bounding box
#             x1, y1, x2, y2 = box.xyxy[0]
#             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

#             # put box in cam
#             cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

#             # confidence
#             confidence = math.ceil((box.conf[0]*100))/100
#             print("Confidence --->",confidence)

#             # class name
#             cls = int(box.cls[0])
#             print("Class name -->", classNames[cls])

#             # object details
#             org = [x1, y1]
#             font = cv2.FONT_HERSHEY_SIMPLEX
#             fontScale = 1
#             color = (255, 0, 0)
#             thickness = 2

#             cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

#     cv2.imshow('Webcam', img)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



# from ultralytics import YOLO
# import cv2
# import math
# import numpy as np 

# # Function to detect people using YOLOv8
# def detect_people(img):
#     height, width, _ = img.shape
#     blob = cv2.dnn.blobFromImage(img, 1/255.0, (416, 416), (0, 0, 0), True, crop=False)
#     net.setInput(blob)
#     output_layers_names = net.getUnconnectedOutLayersNames()
#     layer_outputs = net.forward(output_layers_names)

#     boxes = []
#     confidences = []
#     class_ids = []

#     for output in layer_outputs:
#         for detection in output:
#             scores = detection[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]
#             if confidence > 0.5 and class_id == 0:  # 0 corresponds to 'person' class in COCO dataset
#                 center_x = int(detection[0] * width)
#                 center_y = int(detection[1] * height)
#                 w = int(detection[2] * width)
#                 h = int(detection[3] * height)
#                 x = int(center_x - w / 2)
#                 y = int(center_y - h / 2)
#                 boxes.append([x, y, w, h])
#                 confidences.append(float(confidence))
#                 class_ids.append(class_id)

#     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
#     font = cv2.FONT_HERSHEY_PLAIN
#     colors = np.random.uniform(0, 255, size=(1, 3))

#     for i in range(len(boxes)):
#         if i in indexes:
#             x, y, w, h = boxes[i]
#             color = colors[0]
#             cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(img, 'Person', (x, y + 30), font, 3, color, 3)
    
#     return img

# # Start webcam
# cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# cap.set(3, 1280)
# cap.set(4, 720)

# # Load YOLOv8 model
# net = cv2.dnn.readNet('yolo-Weights/yolov8n.pt')

# # Object classes
# classNames = ["person"]

# while True:
#     success, img = cap.read()

#     # Detect people in the frame
#     img_detected = detect_people(img)

#     # Display frame with detected people
#     cv2.imshow('Webcam', img_detected)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()












