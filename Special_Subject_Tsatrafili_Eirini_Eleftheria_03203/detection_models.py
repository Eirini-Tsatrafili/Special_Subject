import cv2
import torch
import glob
import os
import urllib.request
import time
import csv
import torchvision
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Function to visualize detections
def visualize_detections(model, image_path, detections, output_dir, original_width, original_height, confidence_threshold=0.7, class_names=None):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return
    
    # Convert BGR to RGB for visualization
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process detections
    if(model == "yolov5s"):
        boxes = detections[:, :4]  # Bounding boxes (x1, y1, x2, y2)
        scores = detections[:, 4]  # Confidence scores
        labels = detections[:, 5].astype(int)  # Class labels
    else :
        # Process detections
        boxes = detections[0]['boxes'].cpu().numpy()  # Bounding boxes
        labels = detections[0]['labels'].cpu().numpy()  # Class labels
        scores = detections[0]['scores'].cpu().numpy()  # Confidence scores

    # Calculate the scaling factors
    scale_x = original_width / 800
    scale_y = original_height / 800

    class_counts = 0

    # class_counts = {class_name: 0 for class_name in CLASS_NAMES}

    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence_threshold:
            
            # Scale the box coordinates back to the original image size
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)

            if x1 >= 0 and y1 >= 0 and x2 <= image_rgb.shape[1] and y2 <= image_rgb.shape[0]:
                # Draw the bounding box on the original image
                cv2.rectangle(image_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

            text = f"{score:.2f}"

            # Increment the count for this class
            class_counts += 1
            
            # Put text above the bounding box
            cv2.putText(image_rgb, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    # Save the image to the output directory
    output_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_path, cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)) 
    print(f"Saved processed image: {output_path}")

    return class_counts

def fcos_resnet50_fpn():
    model = torchvision.models.detection.fcos_resnet50_fpn(weights=torchvision.models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT)
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move the model to the appropriate device
    return model

def fasterrcnn_mobilenet_v3_large_fpn():
    model = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(weights=torchvision.models.detection.FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    model.eval()
    model.to(device)
    return model

def fasterrcnn_resnet50_fpn_v2():
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    model.eval() 
    model.to(device) 
    return model

def ssdlite320_mobilenet_v3_large():
    model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(weights=torchvision.models.detection.SSDLite320_MobileNet_V3_Large_Weights.DEFAULT)
    model.eval()
    model.to(device)
    return model

def load_yolo_model():
    model = YOLO("yolov5s.pt") 
    return model

def loading_model (model_name):
    if (model_name == "fcos_resnet50_fpn"):
        model_detection = fcos_resnet50_fpn()
    elif (model_name == "fasterrcnn_mobilenet_v3_large_fpn"):
        model_detection = fasterrcnn_mobilenet_v3_large_fpn()
    elif (model_name == "fasterrcnn_resnet50_fpn_v2"):
        model_detection = fasterrcnn_resnet50_fpn_v2()
    elif (model_name == "ssdlite320_mobilenet_v3_large"):
        model_detection = ssdlite320_mobilenet_v3_large()
    elif (model_name == "yolov5s"):
        model_detection = load_yolo_model()
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    output_dir = os.path.join("detections", model_name)
    csv_file_path = os.path.join("testing", f"{model_name}.csv")

    # Ensure the output and CSV directories exist
    os.makedirs(output_dir, exist_ok=True)
    
    return model_detection, output_dir, csv_file_path
        
# Set the device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Preprocessing: Resize the image and normalize it
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),  # Convert the image to a tensor
    torchvision.transforms.Resize((800, 800), antialias=True),  # Resize to a fixed size
])

# Specify the directory containing PNG images
image_dir = 'trafic_data/train/images'  # Replace with your directory path
image_paths = glob.glob(os.path.join(image_dir, '*.jpg'))

# Check if any images were found
if not image_paths:
    print("No PNG images found in the specified directory.")
    exit()

models_list = [ "yolov5s", "fcos_resnet50_fpn", "fasterrcnn_mobilenet_v3_large_fpn", "fasterrcnn_resnet50_fpn_v2", "ssdlite320_mobilenet_v3_large"]
for model in models_list:  
    # Load the SSD-Lite model
    model_detection, output_dir, csv_file_path = loading_model (model)
    
    with open(csv_file_path, mode='w', newline='') as csv_file:
        fieldnames = ['id','Image Name', 'Total Time (s)', 'Preprocessing Time (s)',
                      'Detection Time (s)', 'Classification Time (s)', 'Post-processing Time (s)', ' ', 'Items Detected']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    
        image_count = -1
        for image_path in image_paths:
            # Read the image
            image_count += 1
            print(f"Processing {image_count} of {len(image_path)} {image_path}")
            image_name = os.path.basename(image_path)

            total_start_time = time.perf_counter()
    
            # Start preprocessing time
            preprocessing_start_time = time.perf_counter()
            
            frame = cv2.imread(image_path)
            if frame is None:
                print(f"Failed to load image {image_path}")
                continue
    
            # Get original image dimensions for scaling the boxes
            original_height, original_width = frame.shape[:2]
            
            # Convert frame to PIL Image
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            #image = Image.open(frame).convert("RGB")
    
            # Transform the image for detection
            img_tensor = transform(image).to(device)
            img_tensor = img_tensor.unsqueeze(0)
    
            # End preprocessing time
            preprocessing_end_time = time.perf_counter()
            preprocessing_time = preprocessing_end_time - preprocessing_start_time
    
            # Start detection time
            detection_start_time = time.perf_counter()

            if(model == "yolov5s"):
                results = model_detection(img_tensor)
                if isinstance(results, list):
                    results = results[0] 
                detections = results.boxes.data.cpu().numpy()   # Get detections (x1, y1, x2, y2, score, class)
            else:
                # Perform detection
                with torch.no_grad():
                    detections = model_detection(img_tensor)
        
            # End detection time
            detection_end_time = time.perf_counter()
            detection_time = detection_end_time - detection_start_time
    
            # Post-processing detections
            postprocessing_start_time = time.perf_counter()

            total_detections = visualize_detections(model, image_path, detections, output_dir, original_width, original_height)

            postprocessing_end_time = time.perf_counter()
            postprocessing_time = postprocessing_end_time - postprocessing_start_time
    
            # Calculate total time
            total_end_time = time.perf_counter()
            total_time = total_end_time - total_start_time

            # Update the header dynamically to include class names with non-zero counts
            dynamic_header = fieldnames #+ [class_name for class_name in filtered_class_counts]
        
            # Write dynamic header (you should write this once, at the start of the file)
            writer = csv.DictWriter(csv_file, fieldnames=dynamic_header)
        
            # Write the header to the CSV file (only once)
            writer.writeheader()
    
            # Write data to the CSV file
            row = {
                'id': image_count,
                'Image Name': image_name,
                'Total Time (s)': total_time,
                'Preprocessing Time (s)': preprocessing_time,
                'Detection Time (s)': detection_time,
                'Post-processing Time (s)': postprocessing_time,
                'Items Detected': total_detections
            }
            
            # Write the row to the CSV file
            writer.writerow(row)

            print(f"Detection results saved to {csv_file_path}.")