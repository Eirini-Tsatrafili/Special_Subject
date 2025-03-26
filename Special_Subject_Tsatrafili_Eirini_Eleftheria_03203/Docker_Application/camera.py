import cv2
import argparse
import numpy as np
import pyrealsense2 as rs
import gi
import argparse

# Import required library like Gstreamer and GstreamerRtspServer
gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GObject

# Sensor Factory class which inherits the GstRtspServer base class and add
# properties to it.
class SensorFactory(GstRtspServer.RTSPMediaFactory):
    
    def __init__(self, opt, **properties):
        super(SensorFactory, self).__init__(**properties)
        print("SensorFactory initialization!")

        # Ensure `opt.ip` is treated as a string correctly
        if str(opt.ip) == '10.64.83.237':
            
            # Configure color streams
            print("🎥 Setting up RealSense camera...")
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            
            
            # Force specific stream configurations
            self.config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)
            self.config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)  
            
            # Get device product line for setting a supporting resolution
            self.pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
            self.device = self.pipeline_profile.get_device()
            self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))

            self.found_rgb = False
            for s in self.device.sensors:
                if s.get_info(rs.camera_info.name) == 'RGB Camera':
                    self.found_rgb = True
                    break
            if not self.found_rgb:
                print("The demo requires Depth camera with Color sensor")
                exit(0)

            # Start streaming
            self.profile = self.pipeline.start(self.config)

        self.number_frames = 0
        self.fps = opt.fps
        self.duration = 1 / self.fps * Gst.SECOND  # duration of a frame in nanoseconds

        self.launch_string = 'appsrc name=source is-live=true block=true format=GST_FORMAT_TIME ' \
                     'caps=video/x-raw,format=BGR,width={},height={},framerate={}/1 ' \
                     '! videoconvert ! video/x-raw,format=I420 ' \
                     '! x264enc speed-preset=ultrafast tune=zerolatency ' \
                     '! rtph264pay config-interval=1 name=pay0 pt=96' \
                     .format(opt.image_width, opt.image_height, opt.fps)

    # Closes pipeline of RealSense
    def __del__(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()  # self.config

    # Method to capture the video feed and push it to the streaming buffer.
    def on_need_data(self, src, length):


        print("Data Request triggered, requesting frame...")
        frame_resize = None  # Initialize frame variable

        if self.pipeline:

            print("Checking if RealSense pipeline is running...")
            if not hasattr(self, 'pipeline') or self.pipeline is None:
                print("RealSense pipeline is not set up!")

            frame = self.pipeline.wait_for_frames(timeout_ms=15000)
            if frame:
                print("Frame retrieved successfully!")
            else:
                print("No frame received!")

            color_frame = frame.get_color_frame()

            if color_frame:
                print("Received color frame")
                color_image = np.asanyarray(color_frame.get_data())
                #frame_resize = cv2.resize(color_image, (640, 480))
                
                print(f"Color image shape: {color_image.shape}")
                print(f"Frame preview (top-left corner pixel): {color_image[0, 0]}")


        if frame_resize is None:
            print("No frame available! Sending black frame instead.")
            frame_resize = np.zeros((480, 640, 3), dtype=np.uint8)

        # Push frame to GStreamer pipeline
        data = frame_resize.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        buf.duration = self.duration
        timestamp = self.number_frames * self.duration
        buf.pts = buf.dts = int(timestamp)
        buf.offset = timestamp
        self.number_frames += 1
        retval = src.emit('push-buffer', buf)
        print(f'Pushed frame {self.number_frames}, duration {self.duration} ns, durations {self.duration / Gst.SECOND} s')

        if retval != Gst.FlowReturn.OK:
            print(f"GStreamer push-buffer error: {retval}")

    # Attach the launch string to the override method
    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)
    
    # Attaching the source element to the rtsp media
    def do_configure(self, rtsp_media):

        print("RTSP configuration")
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)
        
        print("Attaching need-data signal to appsrc")


# Rtsp server implementation where we attach the factory sensor with the stream uri
class GstServer(GstRtspServer.RTSPServer):
    
    def __init__(self, opt, **properties):
        
        super(GstServer, self).__init__(**properties)
        print("Creating SensorFactory...")
        self.factory = SensorFactory(opt)
        self.factory.__init__(opt)
        self.factory.set_shared(True)
        print("SensorFactory created. Setting up RTSP server...")

        # Bind RTSP server to IP and port
        self.set_service(str(opt.port))
        self.get_mount_points().add_factory(opt.stream_uri, self.factory)
        print("Factory attached to RTSP server on URI:", opt.stream_uri)
        
        self.attach(None)

# Argument Parsing
def parse_args():

    # Getting the required information from the user 
    parser = argparse.ArgumentParser(description="Stream Video with RTSP")
    parser.add_argument("--device_id", required=True, help="device id for the \
                    video device or video file location")
    parser.add_argument("--fps", required=True, help="fps of the camera", type = int)
    parser.add_argument("--image_width", required=True, help="video frame width", type = int)
    parser.add_argument("--image_height", required=True, help="video frame height", type = int)
    parser.add_argument("--ip", required=True, help="last digits of ip of the server")
    parser.add_argument("--port", required=True, help="port to stream video", type = int)
    parser.add_argument("--stream_uri", default="/video_stream", help="rtsp video stream uri")
    opt = parser.parse_args()

    return opt

def main():
    opt = parse_args()

    print("Initializing GStreamer...")
    Gst.init(None)

    print("Starting the RTSP server...")
    server = GstServer(opt)
    print("RTSP server is running!")

    loop = GObject.MainLoop()
    loop.run()


if __name__ == "__main__":
     main()    