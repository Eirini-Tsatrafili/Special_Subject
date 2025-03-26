import cv2
import torch
import os
import time
import csv
import argparse
from PIL import Image
import numpy as np
from ultralytics import YOLO
import torchvision
import torchvision.transforms as T

# Function to visualize detections (modifying it to accept frame input)
def visualize_detections(frame, detections, output_dir, original_width, original_height, confidence_threshold=0.2, class_names=None):
    
    # Convert BGR to RGB for visualization
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process detections
    boxes = detections[:, :4]  # Bounding boxes (x1, y1, x2, y2)
    scores = detections[:, 4]  # Confidence scores
    labels = detections[:, 5].astype(int)  # Class labels

    # Calculate the scaling factors
    scale_x = original_width / 640
    scale_y = original_height / 640

    class_counts = 0

    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence_threshold:
            # Scale the box coordinates back to the original image size
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

            if x1 >= 0 and y1 >= 0 and x2 <= image_rgb.shape[1] and y2 <= image_rgb.shape[0]:
                # Draw the bounding box on the original image
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Get the class label for the current object
            label_text = class_names[label] if class_names else f"Class {label}"

            # Add label and score text above the bounding box
            text = f"{label_text} ({score:.2f})"
            class_counts += 1
            
            # Put text above the bounding box
            cv2.putText(image_rgb, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    # Save the image to the output directory
    output_path = os.path.join(output_dir, f"detected_{time.time()}.jpg")
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)) 
    print(f"Saved processed image: {output_path}")

    return class_counts

# Load YOLO model (with pre-trained weights)
def load_yolo_model():
    model = YOLO("yolov5s.pt") 
    return model

# Prepare the model and directories
def loading_model():
    model_detection = load_yolo_model()
    output_dir = os.path.join("detections", "yolov5")
    csv_file_path = os.path.join("testing", "yolov5.csv")

    os.makedirs(output_dir, exist_ok=True)
    csv_dir = os.path.dirname(csv_file_path)
    os.makedirs(csv_dir, exist_ok=True)
    
    return model_detection, output_dir, csv_file_path

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO RTSP Object Detection")
    parser.add_argument("--rtsp_url", type=str, required=True, help="RTSP stream URL")
    return parser.parse_args()

def main():

    args = parse_args()

    # Set the device to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Preprocessing: Resize the image and normalize it
    transform = T.Compose([
        T.Resize((640, 640)),  # Resize to a fixed size for YOLO
        T.ToTensor(),
    ])

    # Initialize RTSP stream
    feed1 = cv2.VideoCapture("rtsp://10.64.83.237:8554/video_stream")
    print(f"📡 Connecting to RTSP stream: {args.rtsp_url}")
    feed = cv2.VideoCapture(args.rtsp_url)

    # Ensure the feed is opened
    if not feed1.isOpened():
        print("Failed to open RTSP stream.")
        exit()

    model_detection, output_dir, csv_file_path = loading_model()

    # Open the CSV file for writing
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = ['id','Image Name', 'Total Time (s)', 'Preprocessing Time (s)',
                    'Detection Time (s)', 'Post-processing Time (s)', 'Items Detected']
        writer = csv.DictWriter(csv_file, fieldnames = fieldnames)

        writer.writeheader()  # Write the header only once
        
        image_count = 0
        while feed1.isOpened():
            ret, frame = feed1.read()
            if not ret:
                print("Failed to read frame from RTSP stream.")
                break
            
            # Display grey-filtered videos
            grey1 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cv2.imshow("RealSense", grey1)

            image_name = f"frame_{image_count}"
            image_count += 1

            total_start_time = time.perf_counter()

            # Start preprocessing time
            preprocessing_start_time = time.perf_counter()
            
            original_height, original_width = frame.shape[:2]
            
            # Convert frame to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            # Transform the image for detection
            img_tensor = transform(image).unsqueeze(0).to(device)
            # img_tensor = transform(image).to(device)
            # img_tensor = img_tensor.unsqueeze(0)

            # End preprocessing time
            preprocessing_end_time = time.perf_counter()
            preprocessing_time = preprocessing_end_time - preprocessing_start_time

            # Start detection time
            detection_start_time = time.perf_counter()

            # Perform detection
            results = model_detection(img_tensor)
            if isinstance(results, list):
                results = results[0] 
            detections = results.boxes.data.cpu().numpy()   # Get detections (x1, y1, x2, y2, score, class)
            for det in detections:
                x1, y1, x2, y2, conf, cls = det  # Unpack values
                print(f"Detected class {cls} with confidence {conf:.2f}")

            # End detection time
            detection_end_time = time.perf_counter()
            detection_time = detection_end_time - detection_start_time

            # Post-processing detections
            postprocessing_start_time = time.perf_counter()

            # Get class names from the model's predefined classes
            class_names = results.names  # This is already a dictionary with class names

            total_detections = visualize_detections(frame, detections, output_dir, original_width, original_height, class_names=class_names)

            postprocessing_end_time = time.perf_counter()
            postprocessing_time = postprocessing_end_time - postprocessing_start_time

            # Calculate total time
            total_end_time = time.perf_counter()
            total_time = total_end_time - total_start_time

            # Write the row to the CSV file
            row = {
                'id': image_count,
                'Image Name': image_name,
                'Total Time (s)': total_time,
                'Preprocessing Time (s)': preprocessing_time,
                'Detection Time (s)': detection_time,
                'Post-processing Time (s)': postprocessing_time,
                'Items Detected': total_detections
            }

            writer.writerow(row)

            print(f"Processed frame {image_count}, saved results to CSV.")

            # Display the frame (optional)
            cv2.imshow("Detected Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the feed and close windows
    feed1.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

