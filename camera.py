import pyrealsense2 as rs
import rsData
from rsData import EStateMode
import numpy as np
import cv2
from PySide2 import QtWidgets,QtCore
class CamHandler(QtCore.QObject):

    def __init__(self):
        QtCore.QObject.__init__(self)
        self.pipeline = None
        self.config = None
        self.depth_scale = 0.0
        self.initilized_cam = False
        self.init_cam()

    def init_cam(self):
        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()
            self.pipeline_wrapper = rs.pipeline_wrapper(rs.pipeline())
            self.pipeline_profile = self.config.resolve(self.pipeline_wrapper)
            self.device = self.pipeline_profile.get_device()
            self.device_product_line = str(self.device.get_info(rs.camera_info.product_line))
            self.camera_info = rs.camera_info
            self.stream = rs.stream
            self.format = rs.format
            self.initilized_cam = True
            rsData.state[0] = EStateMode.eNormal
        except Exception as e:
            self.initilized_cam = False
            rsData.state[0] = EStateMode.eErr_Critical
            # message = QtWidgets.QMessageBox.critical(self, "python", "init_cam Failed!")

    def config_device(self):
        try:
            self.config.enable_stream(self.stream.depth, 640, 480, rs.format.z16, 30)
            if self.device_product_line == 'L500':
               self.config.enable_stream(self.stream.color, 960, 540, rs.format.bgr8, 30)
            else:
               self.config.enable_stream(self.stream.color, 640, 480, rs.format.bgr8, 30)

            # Start streaming
            self.pipeline.start(self.config)
            self.depth_scale = self.device.first_depth_sensor().get_depth_scale()
        except Exception as e:

            message = QtWidgets.QMessageBox.critical(self, "python", str(e))

    def getIntrinsicArray(self):
        profile = self.pipeline.get_active_profile()
        # print('profile',profile)
        color_profile = rs.video_stream_profile(profile.get_stream(rs.stream.color))
        color_intrinsics = color_profile.get_intrinsics()

        return [color_intrinsics.fx, 0, color_intrinsics.ppx, 0, color_intrinsics.fy, color_intrinsics.ppy, 0, 0, 1]

    def stop(self):
        if self.pipeline:
            self.pipeline.stop()

    def streaming(self):
        try:
            if not self.initilized_cam:
                self.init_cam()
            if not self.initilized_cam:
                rsData.log.critical("initilize cam Failed!")
                return
            self.config_device()
            while True:
                if rsData.b_stop_streaming:
                    # cv2.destroyAllWindows()
                    break
                # Wait for a coherent pair of frames: depth and color
                frames = self.pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
                depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

                depth_colormap_dim = depth_colormap.shape
                color_colormap_dim = color_image.shape

                # If depth and color resolutions are different, resize color image to match depth image for display
                if depth_colormap_dim != color_colormap_dim:
                    resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                     interpolation=cv2.INTER_AREA)
                    images = np.hstack((resized_color_image, depth_colormap))
                else:
                    images = np.hstack((color_image, depth_colormap))

                rsData.depthImg = depth_image
                # rsData.colorImg = color_image
                rsData.color_streaming = color_image
                # Show images
                # cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
                # cv2.imshow('RealSense', images)
                # cv2.waitKey(1)
        except Exception as e:
            rsData.log.critical(e)
        # finally:
        #     # Stop streaming
        #     if self.pipeline:
        #         self.pipeline.stop()
