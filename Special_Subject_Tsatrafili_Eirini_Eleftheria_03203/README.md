Development and evaluation of a lab-based smart city testbed with dynamic reconfiguration capabilities

This project focuses on developing and evaluating a lab-based testbed to assess various object detection methods. The testbed consists of an Intel RealSense camera for real-time video capture, paired with a Raspberry Pi as the primary processing unit. The system was used to test and compare the performance of different object detection models in a lab-based environment. The system also incorporates an RTSP server to stream live video, enabling easy remote monitoring and control of the testbed. The entire system is deployed within a Docker container, allowing for simplified setup, testing, and deployment.


-------------------------------------------------------------------------------------------------------------------------------


The detection_models.py contains the code that tests the performance of the models.
- it uses the dataset trafic_data
- no video streaming needed


The other 4 files in the Docker_Application, camera.py, detector.py, Dockerfile and requirements.txt make the application.

Camera.py Features
------------------
Video Streaming: Captures video from a RealSense camera and streams it in real-time.
RTSP Protocol: Streams the video feed over RTSP, allowing easy access via RTSP players like VLC.
GStreamer Integration: Uses GStreamer for video encoding and streaming.
Configuration: Allows custom FPS, width, height, ip and port 

Camera.py Requirements
----------------------
Intel RealSense Camera: The camera is supported by the pyrealsense2 library.
Python 3: The code has been tested with Python 3.
GStreamer: Required for the video streaming over RTSP.
pyrealsense2: Required for the connection with the RealSense camera.
GObject: Required by GStreamer for signal handling.

--------------------------------------------------------------------------------------------

Detector.py Features
--------------------

RTSP Connection: Connecting to the live stream.
Image Processing: Resizes and normalizes images for model compatibility.
Detection Visualization: Draws bounding boxes and confidence scores on detected objects.
Performance Logging: Logs detection times and number of objects detected to CSV.

Detector.py Requirements
------------------------
PyTorch: For running the detection models.
Torchvision: Provides pre-trained detection models.
Ultralytics YOLO: Library for running YOLO models.
OpenCV: For image manipulation and visualization.
Pillow: For handling image formats.
Matplotlib: For potential visualizations.

--------------------------------------------------------------------------------------------

Dockerfile Features
-------------------
Lightweight Container: Uses a Python image.
Installations: Installs packages from requirements.txt and all the dependencies.
Environment Setup: Copies application files and sets working directory.


Usage
-----

docker build -t realsense-rtsp .

docker run -it --rm --privileged --device /dev/video24:/dev/video24 --device
/dev/bus/usb:/dev/bus/usb --device-cgroup-rule=’c 81:* rmw’ -p 8554:8554
--memory="4g" realsense-rtsp bash


! make sure that the ip address is correct to the device for the camera.py and the detector.py code.
Example of running the application:

source /opt/venv/bin/activate

python camera.py --device id 0 --fps 30 --image width 640 --image height 480
--ip 10.64.83.237 --port 8554 --stream uri /video stream.

python detector.py 

----------------------------------------------------------------------------------------------

Running the code without Dockerfile 

-Download the drivers -> commands in file Application/driver.txt

ssh -L 8888:localhost:8888 <host_name>@<host_ip> -X

source myenv/bin/activate

jupyter-notebook --no-browser --ip=0.0.0.0 --port=8888

-use the interface to run the codes in the Application folder -> same files in jupyter notebook format

-check for the rtsp stream by using VLC