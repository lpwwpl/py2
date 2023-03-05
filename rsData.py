import numpy as np
from PySide2 import QtCore
colorImg=np.zeros((480,640,3), np.uint8)
depthImg=np.zeros((480,640), np.uint16)
inferImg=np.zeros((480,640,3), np.uint8)
clusters=np.zeros((480,640,3), np.uint8)
depth_scale = 0.001
axisPoints = np.asarray([])
axisPointsColor = np.asarray([])
camPointsColor = np.asarray([])
camPointsColor_clusters = np.asarray([])
color_streaming=np.zeros((480,640,3), np.uint8)
depth_streaming=np.zeros((480,640,3), np.uint8)

torch_device = None
net = None
softmax = None
r = None
r_thread = QtCore.QThread()
import logging
from enum import Enum
class EMode(Enum):
    # 为序列值指定value值
    eCali = 1
    eRuntime = 2
    eReview = 3
class EStateMode(Enum):
    eNormal = 1
    eErr_Critical=2
    eErr_Important = 3
    eWarning = 4
class ERunning(Enum):
    eRunning=1
    eNotRunning=2
r_ready = False
log = logging.getLogger('vision')
running = ERunning.eNotRunning
mode = EMode.eRuntime
state = [EStateMode.eNormal,EStateMode.eNormal]
robotPos = []
curJoints = []
need_new_folder = False
pc1 = np.zeros((480,640,3),np.uint8)
pc2 = np.zeros((480,640,3),np.uint8)
root_path = ""
cur_root_path=""
width=640
height=480
b_stop_streaming = True
m_themes = {}
m_style = "Light"
clients = []
server = None
b_ws_server_stop = False
url = ""