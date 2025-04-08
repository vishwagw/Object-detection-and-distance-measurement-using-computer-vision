# main program:
# importing libs:
import cv2
import numpy as np
import torch
from time import time
from ultralytics import YOLO

# loading a yolo model:
# using yolov8-nano model:
# class for distance calculation:
class distanceMeasurement:
    def __init__(self):
        # initializing yolo model:
        self.model = YOLO('./yolov10n.pt')

        # Known parameters - these would be calibrated for your specific setup
        # Focal length (pixels) = (image width in pixels) * (real width / apparent width)
        self.focal_length = 1000  # Example value, needs to be calibrated
        
        # Known width of objects in inches/cm
        self.KNOWN_WIDTH = {
            'person': 60,  # average width in cm
            'car': 180,    # average width in cm
            'bottle': 8,   # average width in cm
            'laptop': 35   # average width in cm
        }
        
        # Default width for objects not in our dictionary
        self.DEFAULT_WIDTH = 30  # cm

    # function for calculating distance:
    def calculate_distance(self, bbox_width, object_class):
        """
        Calculate distance based on known object width and apparent width in image
        Using the formula: distance = (known width × focal length) / apparent width
        """
        real_width = self.KNOWN_WIDTH.get(object_class, self.DEFAULT_WIDTH)
        distance = (real_width * self.focal_length) / bbox_width
        return distance
    
    # processing result frame:
    def process_frame(self, frame):
        """Process a single frame, detect objects and estimate distances"""
        # Make a copy of the frame to draw on
        result_frame = frame.copy()
        
        # Run object detection with YOLOv8
        results = self.model(frame)
        
        # Process each detection from the first (and only) image result
        for r in results:
            boxes = r.boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Get confidence score
                conf = box.conf[0].item()
                
                # Get class name
                class_id = int(box.cls[0].item())
                class_name = self.model.names[class_id]
                
                if conf < 0.5:  # Filter low confidence detections
                    continue
                
                # Calculate bbox width
                bbox_width = x2 - x1
                
                # Calculate distance
                distance = self.calculate_distance(bbox_width, class_name)
                
                # Draw bounding box
                cv2.rectangle(result_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                
                # Draw label with class name and distance
                label = f"{class_name}: {distance:.2f} cm"
                cv2.putText(result_frame, label, (int(x1), int(y1) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return result_frame
    
    # calibrate focal length:
    def calibrate_focal_length(self, known_distance, known_width, apparent_width):
        """
        Calibrate the focal length using an object of known size at a known distance
        focal_length = (apparent width * known distance) / known width
        """
        self.focal_length = (apparent_width * known_distance) / known_width
        print(f"Focal length calibrated to: {self.focal_length}")
        return self.focal_length
    
    # process the final output
    def process_video(self, video_source='./input1.mp4'):
        """Process video from file or webcam"""
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source.")
            return
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Create output video writer if source is not webcam
        if video_source != 0:
            output_path = 'output_distance_detection.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame = cv2.resize(frame, (640, 480))  # Resize frame for processing
            # Process the frame
            start_time = time()
            result_frame = self.process_frame(frame)
            end_time = time()
            
            # Calculate and display FPS
            processing_time = end_time - start_time
            fps_text = f"FPS: {1/processing_time:.2f}" if processing_time > 0 else "FPS: N/A"
            cv2.putText(result_frame, fps_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display the result
            cv2.imshow("Object Detection with Distance Measurement", result_frame)
            
            # Write frame to output video if not using webcam
            if video_source != 0:
                out.write(result_frame)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        cap.release()
        if video_source != 0:
            out.release()
        cv2.destroyAllWindows()

# initializing the program:
if __name__ == "__main__":
    distance_detector = distanceMeasurement()
    
    # Optional: Calibrate with a known object at a known distance
    # Example: A person (60cm wide) at 300cm distance appears 200 pixels wide
    # distance_detector.calibrate_focal_length(known_distance=300, known_width=60, apparent_width=200)
    
    # Process video
    video_source = './input4.mp4'  # Use 0 for webcam or a path to a video file
    distance_detector.process_video(video_source)
