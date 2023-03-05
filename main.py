from abb import *
import cv2
import sys
from PySide2 import QtWidgets
from PySide2 import QtCore,QtGui
import rsData 
import threading
from confBLL import ConfBLL
import os
from pathlib import Path
import glob
import re
from view import Kawasaki_lx
from view import KCore
from rsData import EMode
from rsData import ERunning
from rsData import EStateMode
timestamp = None
from ModeControlTabWidget import CaliActionWidget,CaliParamActionWidget
from camera import CamHandler
from logview import LogView
from ModeControlTabWidget import ModeControlTabWidget
from QFileSystemModel.filesystem_treeview import FileSystemWidget
from about import About
from UtilSet import check_ip
from robotQml import *
from OpenGLControl import *
from kafka_me import kfkProducer
from PySide2.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PySide2.QtWebChannel import QWebChannel
from kafka import KafkaProducer
from PySide2.QtWebChannel import QWebChannel
from PySide2.QtWebSockets import QWebSocketServer
from PySide2.QtNetwork import QHostAddress
from SimpleWebSocketServer import *
kafka_mgr = {
    "broker" : '127.0.0.1',
    "port" : 9092,
}


def executable_path():
    return os.path.dirname(sys.argv[0])


class IPInputQwidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.initUI()

    def initUI(self):
        self.lineEdit = QtWidgets.QLineEdit()
        self.layout = QtWidgets.QGridLayout()
        self.layout.addWidget(self.lineEdit)
        self.setLayout(self.layout)


class ParamWidget(QtWidgets.QWidget):
    stylesheet_signal = QtCore.Signal()
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.initUI()

    def initUI(self):
        self.label_robot = QtWidgets.QLabel("RobotType:")
        self.label_robot_ip = QtWidgets.QLabel("RobotIp:")
        self.cb_robot = QtWidgets.QComboBox()
        self.cb_robot.addItem("ABB")
        self.cb_robot.addItem("UR")
        self.ipInput = IPInputQwidget()
        self.layout = QtWidgets.QGridLayout()

        robotIp = ConfBLL.readconf("robot", "robotip")
        robotType = ConfBLL.readconf("robot", "robottype")
        self.ipInput.lineEdit.setText(robotIp)
        self.cb_robot.setCurrentText(robotType)


        self.label_root_path = QtWidgets.QLabel("RootPath:")
        self.label_write_pose = QtWidgets.QLabel("write_pose:")
        self.label_view_after_capture = QtWidgets.QLabel("View after capture::")
        self.linePath = QtWidgets.QTextEdit()
        self.path_chooser = QtWidgets.QPushButton("Choose")
        self.path_chooser.clicked.connect(self.choose)

        self.chk_write_pose = QtWidgets.QCheckBox()
        self.chk_view_after_capture = QtWidgets.QCheckBox()

        self.label_style = QtWidgets.QLabel("Style:")
        self.cmb_style = QtWidgets.QComboBox()
        self.cmb_style.blockSignals(True)
        self.cmb_style.addItem("Light")
        self.cmb_style.addItem("Dark")
        value = ConfBLL.readconf("param","stylesheet")
        self.cmb_style.setCurrentText(value)
        self.cmb_style.blockSignals(False)


        self.label_robot_3d_addr = QtWidgets.QLabel("URL")
        self.line_robot_3d_addr = QtWidgets.QTextEdit()
        self.layout = QtWidgets.QGridLayout()
        # self.layout.setRowStretch(0, 1)
        self.layout.addWidget(self.label_robot, 1, 0, 1, 1)
        self.layout.addWidget(self.cb_robot, 1, 1, 1, 1)
        self.layout.addWidget(self.label_robot_ip, 2, 0, 1, 1)
        self.layout.addWidget(self.ipInput, 2, 1, 1, 1)
        self.layout.addWidget(self.label_root_path, 3, 0, 1, 1)
        self.layout.addWidget(self.linePath, 3, 1, 1, 1)
        self.layout.addWidget(self.path_chooser, 3, 2, 1, 1)
        self.layout.addWidget(self.label_write_pose, 4, 0, 1, 1)
        self.layout.addWidget(self.chk_write_pose, 4, 1, 1, 1)
        self.layout.addWidget(self.label_view_after_capture, 5, 0, 1, 1)
        self.layout.addWidget(self.chk_view_after_capture, 5, 1, 1, 1)
        self.layout.addWidget(self.label_style,6,0,1,1)
        self.layout.addWidget(self.cmb_style,6,1,1,1)
        self.layout.addWidget(self.label_robot_3d_addr,7,0,1,1)
        self.layout.addWidget(self.line_robot_3d_addr,7,1,1,1)
        self.layout.setRowStretch(8, 1)
        self.setLayout(self.layout)

        self.linePath.setText(ConfBLL.readconf("param","root_path"))
        self.line_robot_3d_addr.setText(ConfBLL.readconf("param","url"))

    def choose(self):
        selectedDir = QtWidgets.QFileDialog.getExistingDirectory(self,"Open Directory",executable_path(),QtWidgets.QFileDialog.ShowDirsOnly)
        self.linePath.setText(selectedDir)


    def saveSetting(self):
        value = True
        write_pose = '0'
        if self.chk_write_pose.isChecked():
            write_pose = '1'
        cache_value = ConfBLL.readconf("capture","write_pose")
        if cache_value != write_pose:
            value = value & ConfBLL.writeconf("capture", "write_pose", write_pose)

        view_after_capture = '0'
        if self.chk_view_after_capture.isChecked():
            view_after_capture = '1'
        cache_value = ConfBLL.readconf("capture","view")
        if cache_value!=view_after_capture:
            value = value & ConfBLL.writeconf("capture", "view", view_after_capture)

        selectedDir=self.linePath.document().toPlainText()
        if selectedDir == '':
            return False
        dir = QtCore.QDir(selectedDir)
        cache_value = rsData.root_path
        if dir.exists() and cache_value!= selectedDir:
            value = value & ConfBLL.writeconf("param","root_path",selectedDir)


        cache_value = rsData.url
        url = self.line_robot_3d_addr.document().toPlainText()
        if url != cache_value:
            value = value & ConfBLL.writeconf("param","url",url)

        type = self.cb_robot.currentText()
        ip = self.ipInput.lineEdit.text()
        cache_value = ConfBLL.readconf("robot","robottype")
        if cache_value != type:
            value = ConfBLL.writeconf('robot','robottype',type)
        cache_value = ConfBLL.readconf("robot","robotip")
        if cache_value != ip:
            value = value & ConfBLL.writeconf('robot','robotip',ip)

        style = self.cmb_style.currentText()
        cache_value = ConfBLL.readconf("param","stylesheet")
        if cache_value != style:
            result = ConfBLL.writeconf("param", "stylesheet", style)
            if result:
                rsData.m_style = style
                self.stylesheet_signal.emit()
            value = value & result

        return value

class CaliParamWidget(QtWidgets.QWidget):
    def __init__(self,parent=None) :
        QtWidgets.QWidget.__init__(self, parent)
        self.initUi()

    def reset(self):
        self.cali_editor.reset(rsData.cur_root_path)

    def initUi(self):
        self.cali_editor = CaliActionWidget()
        self.glayout = QtWidgets.QGridLayout()
        self.glayout.addWidget(self.cali_editor,0,0,1,4)
        self.btn_write_back = QtWidgets.QPushButton("WriteBack")
        self.btn_export = QtWidgets.QPushButton("export")
        self.glayout.addWidget(self.btn_write_back,1,2,1,1)
        self.glayout.addWidget(self.btn_export,1,3,1,1,)
        self.btn_write_back.clicked.connect(self.write_back)
        self.btn_export.clicked.connect(self.export)
        self.setLayout(self.glayout)

        # self.cali_editor.reset(rsData.cur_root_path)

    def saveSetting(self):
        return self.export(rsData.cur_root_path)

    def write_back(self):
        curGoalIdx = self.cali_editor.table.currentRow()
        if rsData.r and curGoalIdx >= 0:
            joints = rsData.r.get_joints()
            self.cali_editor.table.item(curGoalIdx, 0).setText(str(joints[0]))
            self.cali_editor.table.item(curGoalIdx, 1).setText(str(joints[1]))
            self.cali_editor.table.item(curGoalIdx, 2).setText(str(joints[2]))
            self.cali_editor.table.item(curGoalIdx, 3).setText(str(joints[3]))
            self.cali_editor.table.item(curGoalIdx, 4).setText(str(joints[4]))
            self.cali_editor.table.item(curGoalIdx, 5).setText(str(joints[5]))

    def export(self,path):
        path ="{}/{}".format(path,"joints.txt")
        fileInfo = QtCore.QFileInfo(path)
        if fileInfo.exists():
            with open(path, 'w+') as f:
                for i in range(self.cali_editor.table.rowCount()):
                    joints0 = self.cali_editor.table.item(i, 0).text()
                    joints1 = self.cali_editor.table.item(i, 1).text()
                    joints2 = self.cali_editor.table.item(i, 2).text()
                    joints3 = self.cali_editor.table.item(i, 3).text()
                    joints4 = self.cali_editor.table.item(i, 4).text()
                    joints5 = self.cali_editor.table.item(i, 5).text()
                    str = "{},{},{},{},{},{}".format(joints0, joints1, joints2, joints3, joints4, joints5)
                    f.write(str)
                return True
            return False
        return True

class RobotParamWidget(QtWidgets.QWidget):
    def __init__(self,parent=None) :
        QtWidgets.QWidget.__init__(self, parent)
        self.initUi()

    def initUi(self):
        self.label_robot = QtWidgets.QLabel("RobotType:")
        self.label_robot_ip = QtWidgets.QLabel("RobotIp:")
        self.cb_robot = QtWidgets.QComboBox()
        self.cb_robot.addItem("ABB")
        self.cb_robot.addItem("UR")
        self.ipInput = IPInputQwidget()
        self.layout = QtWidgets.QGridLayout()

        robotIp = ConfBLL.readconf("robot","robotip")
        robotType = ConfBLL.readconf("robot","robottype")
        self.ipInput.lineEdit.setText(robotIp)
        self.cb_robot.setCurrentText(robotType)
        self.layout.addWidget(self.label_robot,0,0,1,1)
        self.layout.addWidget(self.cb_robot,0,1,1,1)
        self.layout.addWidget(self.label_robot_ip,1,0,1,1)
        self.layout.addWidget(self.ipInput,1,1,1,1)
        # self.layout.setRowStretch(2,1)
        self.setLayout(self.layout)

    def saveSetting(self):
        value = True
        type = self.cb_robot.currentText()
        ip = self.ipInput.lineEdit.text()
        cacheType = ConfBLL.readconf("robot","robottype")
        if cacheType != type:
            value = ConfBLL.writeconf('robot','robottype',type)
        cacheIp = ConfBLL.readconf("robot","robotip")
        if cacheIp != ip:
            value = value & ConfBLL.writeconf('robot','robotip',ip)
        return value

class SetupWidget(QtWidgets.QDialog):
    setup_closed_signal = QtCore.Signal()
    def __init__(self,parent=None) :
        QtWidgets.QDialog.__init__(self, parent)
        self.initUI()

    def closeEvent(self,event):
        self.setup_closed_signal.emit()

    def initUI(self):
        # self.robotParam = RobotParamWidget()
        self.caliParam = CaliParamWidget()
        self.param = ParamWidget()

        self.saveBtn = QtWidgets.QPushButton("Save")
        self.restore = QtWidgets.QPushButton("Restore")
        self.saveBtn.clicked.connect(self.saveSetting)
        self.restore.clicked.connect(self.restoreSetting)
        self.tab = QtWidgets.QTabWidget()
        # self.tab.addTab(self.robotParam,"Robot")
        self.tab.addTab(self.param, "Param")
        self.tab.addTab(self.caliParam,"Cali")


        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.addWidget(self.tab,0,0,5,5)
        self.gridLayout.addWidget(self.saveBtn,5,3,1,1)
        self.gridLayout.addWidget(self.restore,5,4,1,1)
        self.setLayout(self.gridLayout)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)

    def saveSetting(self):
        value = True
        for i in range(self.tab.count()):
            value = value & self.tab.widget(i).saveSetting()

        if value:
            message = QtWidgets.QMessageBox.information(self, "vision", "Save Successful!")
        else:
            message = QtWidgets.QMessageBox.Warning(self, "vision", "Save failed!")

    def restoreSetting(self):
        message = QtWidgets.QMessageBox.Warning(self, "vision", "Nothing!")


class SimpleWSServer(WebSocket):
    def handleConnected(self):
        rsData.clients.append(self)

    def handleClose(self):
        rsData.clients.remove(self)

class RealSenseWidget(QtWidgets.QMainWindow):

    def __init__(self,parent=None) :

        QtWidgets.QMainWindow.__init__(self,parent)
        self.thread_streaming = None
        self.save_dir = None
        self.datafile='data_cali.txt'
        self.tcpfile='data_robotxyzrpy.txt'
        rsData.root_path = ""
        rsData.cur_root_path=""
        self.cam_bll = CamHandler()
        self.initUi()
        self.initTimer()
        self.mode = EMode.eRuntime
        # self.produce = kfkProducer(kafka_mgr["broker"], kafka_mgr["port"], "qml2html_joints")
        # self.produce.__str__()

    def keyPressEvent(self, event) :
        if event.type() == QtCore.QEvent.KeyPress:
            if event.key() == QtCore.Qt.Key_L and event.modifiers() == QtCore.Qt.ControlModifier:
                self.dock.show()
        # QtWidgets.QWidget.keyPressEvent(event)
        return True



    def initTimer(self):
        # self.scan_Timer = QtCore.QTimer()
        # self.scan_Timer.setInterval(30)
        # self.scan_Timer.timeout.connect(self.scan_timeout)

        self.check_timer = QtCore.QTimer()
        self.check_timer.setInterval(30)
        self.check_timer.timeout.connect(self.check_timeout)
        self.check_timer.start()

        self.getRobotInfo_timer = QtCore.QTimer()
        self.getRobotInfo_timer.setInterval(100)
        self.getRobotInfo_timer.timeout.connect(self.getrobotinfo_timeout)
        self.getRobotInfo_timer.start()

    def initMenu(self):
        self.m_viewMenu = self.menuBar().addMenu("View")
        # self.m_fileMenu.addAction(self.)
        self.reviewAction = QtWidgets.QAction("Review",self)
        self.m_viewMenu.addAction(self.reviewAction)

        self.logAction = QtWidgets.QAction("Log",self)
        self.m_viewMenu.addAction(self.logAction)

        self.robot_3d_action = QtWidgets.QAction("Robot3D",self)
        self.m_viewMenu.addAction(self.robot_3d_action)

        self.filetreeAction = QtWidgets.QAction("FileTree",self)
        self.m_viewMenu.addAction(self.filetreeAction)

        self.filetreeAction.setShortcut(QtGui.QKeySequence("Ctrl+F"))
        self.reviewAction.setShortcut(QtGui.QKeySequence("Ctrl+R"))
        self.logAction.setShortcut(QtGui.QKeySequence("Ctrl+L"))
        self.robot_3d_action.setShortcut(QtGui.QKeySequence("Ctrl+T"))

        self.filetreeAction.triggered.connect(self.open_file_tree)
        self.reviewAction.triggered.connect(self.open_review)
        self.logAction.triggered.connect(self.open_log)
        self.robot_3d_action.triggered.connect(self.open_robot_3d)

        self.m_aboutMenu = self.menuBar().addMenu("Help")
        self.aboutAction = QtWidgets.QAction("About", self)
        self.aboutAction.setShortcut(QtGui.QKeySequence("Ctrl+A"))
        self.m_aboutMenu.addAction(self.aboutAction)
        self.aboutAction.triggered.connect(self.about)
        # self.m_editMenu = self.menuBar().addMenu("View")

    def about(self):
        aboutPane = About(self)
        aboutPane.setStyleSheet(rsData.m_themes[rsData.m_style])
        aboutPane.exec_()

    def open_robot_3d(self):
        self.dock_images.show()

    def open_review(self):
        self.dockModeCntl.show()

    def open_file_tree(self):
        self.dockFileTreeCntl.show()

    def open_log(self):
        self.dock_log.show()

    def initToolBar(self):
        self.m_mainToolBar = self.addToolBar("Main Tools")
        self.m_mainToolBar.setIconSize(QtCore.QSize(16,16))
        self.m_mainToolBar.setAllowedAreas(QtCore.Qt.LeftToolBarArea|QtCore.Qt.RightToolBarArea|QtCore.Qt.TopToolBarArea)
        self.m_mainToolBar.setContentsMargins(0,0,0,0)
        self.m_mainToolBar.setFloatable(True)
        self.m_mainToolBar.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
        self.grp_mode = QtWidgets.QGroupBox("")
        self.grp_mode.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                           QtWidgets.QSizePolicy.Preferred)
        # self.grp_mode.setMinimumSize(QtCore.QSize(720,120))
        self.grid_radio  = QtWidgets.QGridLayout()
        self.grid_radio.setContentsMargins(0,0,0,0)
        self.radio_cali = QtWidgets.QRadioButton("Calibration",self.grp_mode)
        self.radio_runtime=QtWidgets.QRadioButton("RunTime",self.grp_mode)
        self.radio_review=QtWidgets.QRadioButton("Review",self.grp_mode)
        self.grid_radio.addWidget(self.radio_runtime,0,0,1,1)
        self.grid_radio.addWidget(self.radio_cali,0,1,1,1)
        self.grid_radio.addWidget(self.radio_review,0,2,1,1)
        self.grp_mode.setLayout(self.grid_radio)
        self.radio_runtime.setChecked(True)
        self.radio_runtime.clicked.connect(self.mode_change_runtime)
        self.radio_review.clicked.connect(self.mode_chage_review)
        self.radio_cali.clicked.connect(self.mode_change_cali)

        self.m_mainToolBar.addWidget(self.grp_mode)

        self.m_mainToolBar.addSeparator()


        self.device_toolBar = QtWidgets.QToolBar("test", self)
        self.device_toolBar.setContentsMargins(0,0,0,0)
        self.addToolBar(QtCore.Qt.TopToolBarArea, self.device_toolBar)
        self.device_toolBar.setAllowedAreas(
            QtCore.Qt.LeftToolBarArea | QtCore.Qt.RightToolBarArea | QtCore.Qt.TopToolBarArea)
        self.device_toolBar.setFloatable(True)
        self.device_toolBar.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
        self.btn_capture = QtWidgets.QPushButton("Capture")
        self.btn_open_abb = QtWidgets.QPushButton("Conn Robot")
        self.btn_setup = QtWidgets.QPushButton("Setup")
        self.btn_streaming_start = QtWidgets.QPushButton("StartScan")
        self.btn_start_runtime = QtWidgets.QPushButton("StartRun")
        self.btn_view = QtWidgets.QPushButton("ViewPCL")

        self.btn_gLayout = QtWidgets.QGridLayout()
        self.btn_gLayout.setContentsMargins(0,0,0,0)
        self.btn_gLayout.addWidget(self.btn_streaming_start,0,0,1,1)
        self.btn_gLayout.addWidget(self.btn_open_abb,0,1,1,1)
        self.btn_gLayout.addWidget(self.btn_capture,0,2,1,1)
        self.btn_gLayout.addWidget(self.btn_view,0,3,1,1)
        self.btn_gLayout.addWidget(self.btn_setup, 0, 4, 1, 1)
        self.btn_gLayout.addWidget(self.btn_start_runtime,0,5,1,1)
        self.btn_gLayout.setColumnStretch(6, 1)
        self.btn_gLayout.setContentsMargins(0, 0, 0, 0)
        self.device_cntl = QtWidgets.QWidget()
        self.device_cntl.setLayout(self.btn_gLayout)
        self.device_toolBar.addWidget(self.device_cntl)
        self.device_toolBar.addSeparator()

        self.m_NavToolBar = QtWidgets.QToolBar("Nav Tools",self)
        self.addToolBar(QtCore.Qt.LeftToolBarArea,self.m_NavToolBar)
        self.m_NavToolBar.setIconSize(QtCore.QSize(24,24))
        self.filetreeAction = QtWidgets.QAction()
        self.filetreeAction.setIcon(QtGui.QIcon("./Resource/images/Camera_20px.png"))
        self.m_NavToolBar.addAction(self.filetreeAction)
        self.m_NavToolBar.setFloatable(True)
        self.m_NavToolBar.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)
        self.m_NavToolBar.setAllowedAreas(
            QtCore.Qt.LeftToolBarArea | QtCore.Qt.RightToolBarArea | QtCore.Qt.TopToolBarArea)


        # self.m_NavToolBar.addAction()

    # def resize

    def initThemeIfExist(self,themeName,path):
        theme = QtCore.QFile(path)
        if theme.exists():
            theme.open(QtCore.QIODevice.ReadOnly|QtCore.QIODevice.Text)
            inf = QtCore.QTextStream(theme)
            styleSheet=inf.readAll()
            rsData.m_themes[themeName] = styleSheet



    def initUi(self):

        self.initThemeIfExist("Dark", "./Resource/dark_style_sheet/qdarkstyle/style.qss")
        self.initThemeIfExist("Light", "./Resource/light_style_sheet/qlightstyle/lightstyle.qss")

        # print(self.m_themes['Dark'])
        rsData.m_style = ConfBLL.readconf("param","stylesheet")
        self.setStyleSheet(rsData.m_themes[rsData.m_style])

        self.setWindowTitle("Vision")


        self.initMenu()
        self.initToolBar()

        write_pose = ConfBLL.readconf('capture', 'write_pose')
        self.write_pose = int(write_pose)

        view = ConfBLL.readconf('capture', 'view')
        self.view_after_capture = int(view)

        
        self.mainLayout = QtWidgets.QGridLayout()
        self.mainLayout.setContentsMargins(0,0,0,0)



        self.lx = Kawasaki_lx()
        self.lx_gLayout = QtWidgets.QGridLayout()
        self.lx_gLayout.setContentsMargins(0,0,0,0)
        self.lx_gLayout.addWidget(self.lx, 1, 1, 1, 1)
        # self.lx_gLayout.setRowStretch(0, 1)
        # self.lx_gLayout.setColumnStretch(0, 1)
        # self.lx_gLayout.setColumnStretch(2, 1)
        self.lx_gLayout.setContentsMargins(0, 0, 0, 0)
        # self.lx_gLayout.setRowStretch(2, 1)
        self.w = QtWidgets.QWidget()
        self.w.setLayout(self.lx_gLayout)
        self.mainLayout.addWidget(self.w,0,0,1,2)


        self.log_widget = LogView()
        ##########################################################
        location = QtCore.Qt.BottomDockWidgetArea
        dock_location = QtCore.Qt.DockWidgetArea(location)
        self.dock_log = QtWidgets.QDockWidget()
        self.dock_log.setWidget(self.log_widget)
        self.dock_log.setAllowedAreas(QtCore.Qt.TopDockWidgetArea|QtCore.Qt.BottomDockWidgetArea)
        self.dock_log.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable|QtWidgets.QDockWidget.DockWidgetClosable|QtWidgets.QDockWidget.DockWidgetFloatable)
        self.dock_log.setWindowTitle("Log")
        self.addDockWidget(dock_location, self.dock_log)
        ##########################################################
        location = QtCore.Qt.LeftDockWidgetArea
        dock_location = QtCore.Qt.DockWidgetArea(location)

        self.fileTreeCntl = FileSystemWidget()

        self.dockFileTreeCntl = QtWidgets.QDockWidget()
        self.dockFileTreeCntl.setWidget(self.fileTreeCntl)
        self.dockFileTreeCntl.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        self.dockFileTreeCntl.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetClosable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(dock_location, self.dockFileTreeCntl)


        self.caliWidget = CaliParamActionWidget()
        self.modeCntl = ModeControlTabWidget()
        self.dockModeCntl = QtWidgets.QDockWidget()
        self.dockModeCntl.setWidget(self.modeCntl)
        self.dockModeCntl.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        self.dockModeCntl.setFeatures(QtWidgets.QDockWidget.DockWidgetMovable|QtWidgets.QDockWidget.DockWidgetClosable|QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(dock_location, self.dockModeCntl)

        ##########################################################


        location = QtCore.Qt.LeftDockWidgetArea
        dock_location = QtCore.Qt.DockWidgetArea(location)
        self.dock_caliWidget = QtWidgets.QDockWidget()
        self.dock_caliWidget.setWidget(self.caliWidget)
        self.dock_caliWidget.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea | QtCore.Qt.RightDockWidgetArea)
        self.dock_caliWidget.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetClosable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(dock_location, self.dock_caliWidget)


        location = QtCore.Qt.RightDockWidgetArea
        dock_location = QtCore.Qt.DockWidgetArea(location)
        # engine = QtQml.QQmlApplicationEngine()
        # engine.load("robot.qml")
        self.qml = RobotQmlWidget()
        # self.dockQml = QtWidgets.QDockWidget()
        # self.dockQml.setWidget(self.qml)
        # self.dockQml.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        # self.dockQml.setFeatures(
        #     QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetClosable | QtWidgets.QDockWidget.DockWidgetFloatable)
        # self.addDockWidget(dock_location, self.dockQml)

        self.objRB = Robot()
        self.RB = GLWidget(None,self.objRB)

        self.setCentralWidget(self.RB)
        # self.setCentralWidget(self.qml)


        self.t = threading.Thread(target=self.run_server)
        self.t.start()



        self.core = KCore(self)
        self.core.pic_signal.connect(self.lx.showImage)

        self.modeCntl.setCore(self.core)
        # self.modeCntl.LoadPiclx()
        self.fileTreeCntl.tree.root_path_signal.connect(self.LoadData)

        self.btn_streaming_start.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_capture.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_open_abb.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_setup.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_view.setFocusPolicy(QtCore.Qt.NoFocus)
        self.btn_start_runtime.setFocusPolicy(QtCore.Qt.NoFocus)

        self.btn_streaming_start.clicked.connect(self.start)
        self.btn_capture.clicked.connect(self.capture)
        self.btn_open_abb.clicked.connect(self.openAbb)
        self.btn_setup.clicked.connect(self.setup)
        self.btn_view.clicked.connect(self.view)
        self.btn_start_runtime.clicked.connect(self.start_runtime)
        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding,
                           QtWidgets.QSizePolicy.Policy.Expanding)

        self.widget = QtWidgets.QWidget(self)
        self.widget.setLayout(self.mainLayout)


        location = QtCore.Qt.RightDockWidgetArea
        dock_location = QtCore.Qt.DockWidgetArea(location)

        self.dock_images = QtWidgets.QDockWidget()
        self.dock_images.setWidget(self.widget)
        self.dock_images.setAllowedAreas(QtCore.Qt.LeftDockWidgetArea|QtCore.Qt.RightDockWidgetArea)
        self.dock_images.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable | QtWidgets.QDockWidget.DockWidgetClosable | QtWidgets.QDockWidget.DockWidgetFloatable)
        self.addDockWidget(dock_location, self.dock_images)




        self.setup_dialog = None
        # self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)

        self.set_root_path()
        self.fileTreeCntl.tree.collapseAll()

        # self.setFocusPolicy(QtCore.Qt.NoFocus)
        # self.dock_log.setMinimumWidth(1080)

    def run_server(self):

        rsData.server = SimpleWebSocketServer('127.0.0.1', 9000, SimpleWSServer,
                                       selectInterval=(1000.0 / 15) / 1000)
        rsData.server.serveforever()

    def set_root_path(self):
        root_path = ConfBLL.readconf("param","root_path")
        if rsData.root_path == root_path:
            return
        rsData.root_path=root_path
        self.fileTreeCntl.set_root_path(root_path)

        # self.LoadData(rsData.root_path)

    def LoadData(self,path):
        # self.closeAbb()
        self.modeCntl.LoadPiclx(path)
        self.caliWidget.reset(path)

    def start_runtime(self):
        try:
            if self.core and self.core.isRunning():
                self.core.working = False
                self.core.wait()

                rsData.robotPos = []
                rsData.curJoints=[]
                self.btn_start_runtime.setText("StartRun")
                self.modeCntl.reviewListWidget.setEnabled(True)
                rsData.running = ERunning.eNotRunning
            else:
                self.core.working = True
                self.core.start()
                self.btn_start_runtime.setText("StopRun")
                self.modeCntl.reviewListWidget.setEnabled(False)
                rsData.running = ERunning.eRunning

        except Exception as e:
            rsData.log.error(e)


    def mode_change_runtime(self):
        self.modeCntl.tab.setCurrentWidget(self.modeCntl.reviewListWidget)
        self.btn_start_runtime.setEnabled(True)
        self.btn_streaming_start.setEnabled(True)
        self.btn_open_abb.setEnabled(True)
        rsData.mode = EMode.eRuntime

    def mode_chage_review(self):
        self.modeCntl.tab.setCurrentWidget(self.modeCntl.reviewListWidget)
        self.btn_start_runtime.setEnabled(False)
        self.btn_streaming_start.setEnabled(False)
        self.btn_open_abb.setEnabled(False)
        rsData.mode = EMode.eReview

    def mode_change_cali(self):
        self.btn_start_runtime.setEnabled(False)
        self.btn_streaming_start.setEnabled(True)
        self.btn_open_abb.setEnabled(True)
        # self.modeCntl.tab.setCurrentWidget(self.modeCntl.caliWidget)
        self.caliWidget.cali.table.setFocus()
        self.mode = EMode.eCali

    def getrobotinfo_timeout(self):
        # pass
        if rsData.r:
            rsData.robotPos = rsData.r.get_cartesian()
        if rsData.r:
            rsData.curJoints = rsData.r.get_joints()

            jsonMsg = rsData.curJoints
            value = []
            for i in range(len(jsonMsg)):
                value.append(str(jsonMsg[i]))
            value=','.join(value)

            # print(value)
            # self.qml.webView.data.jointsChanged.emit(value)

            for client in rsData.clients:
                client.sendMessage(value)
            # self.produce.produceMsg("qml2html_joints", str)


    def check_timeout(self):
        # if rsData.r:
        #     rsData.robotPos = rsData.r.get_cartesian()
        # if rsData.r:
        #     rsData.curJoints = rsData.r.get_joints()
        msg = ""
        if rsData.r and rsData.state[1]==EStateMode.eNormal:
            self.btn_open_abb.setText("Disc Robot")
            # msg = msg + "robot connected"
            self.caliWidget.btn_goto.setEnabled(True)
        else:
            self.btn_open_abb.setText("Conn Robot")
            # msg = msg + "robot disconnected"
            self.caliWidget.btn_goto.setEnabled(False)

        # msg = msg + ", "
        if not self.thread_streaming or not self.thread_streaming.is_alive():
            self.btn_streaming_start.setText("StartScan")
            rsData.b_stop_streaming = True
            # msg = msg + "camera disconnected"
        else:
            self.btn_streaming_start.setText("StopScan")
            # msg = msg + "camera connected"

        if self.core and self.core.isRunning():
            self.btn_start_runtime.setText("StopRun")
        else:
            self.btn_start_runtime.setText("StartRun")

        if rsData.need_new_folder and self.mode == EMode.eCali:
            self.gen_save_dir()
            rsData.need_new_folder = False

        #go_pos
        joints = rsData.curJoints
        if len(joints) == 6:
            msg = msg + "[{}, {}, {}, {}, {}, {}]".format(joints[0],joints[1],joints[2],joints[3],joints[4],joints[5])
        else:
            msg = msg + "[]"

        pose = rsData.robotPos
        if len(pose) == 2:
            pose = rsData.robotPos
            xyz=pose[0]
            cart = pose[1]
            if len(xyz)==3 and len(cart)==4:
                msg = msg + ', ' + "[{}, {}, {}, {}, {}, {}, {}]".format(xyz[0],xyz[1],xyz[2],cart[0],cart[1],cart[2],cart[3])
        else:
            msg = msg + ', ' + "[]"

        curGoalIdx = self.caliWidget.cali.table.currentRow()
        msg = msg + ', ' "{}".format(curGoalIdx+1)
        self.statusBar().showMessage(msg)

    def increment_path(self,path, exist_ok=False, sep='', mkdir=False):
        # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
        path = Path(path)  # os-agnostic
        if path.exists() and not exist_ok:
            suffix = path.suffix
            path = path.with_suffix('')
            dirs = glob.glob(f"{path}{sep}*")  # similar paths
            matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]  # indices
            # n = max(i) + 1
            n = max(i) + 1 if i else 0  # increment number
            path = Path(f"{path}{sep}{n}{suffix}")  # update path
        dir = path if path.suffix == '' else path.parent  # directory
        if not dir.exists() and mkdir:
            dir.mkdir(parents=True, exist_ok=True)  # make directory
        return path

    def gen_save_dir(self, save_dir='runs/calibration_collect/exp'):
        self.save_dir = self.increment_path(save_dir, exist_ok=save_dir != 'runs/calibration_collect/exp', mkdir=True)  # increment save_dir


    def emergy_stop(self):
        if rsData.r:
            rsData.r.stop()

    def cali(self):
        pass
        # self.cali_dialog.show()

    def slotStylesheetChanged(self):
        self.setStyleSheet(rsData.m_themes[rsData.m_style])
        if self.setup_dialog:
            self.setup_dialog.setStyleSheet(rsData.m_themes[rsData.m_style])

    def setup(self):
        self.setup_dialog = SetupWidget()
        self.setup_dialog.param.stylesheet_signal.connect(self.slotStylesheetChanged)
        self.setup_dialog.setStyleSheet(rsData.m_themes[rsData.m_style])
        self.setup_dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setup_dialog.setMinimumSize(QtCore.QSize(720, 480))
        ret =self.setup_dialog.setup_closed_signal.connect(self.set_root_path)
        self.setup_dialog.caliParam.reset()
        self.setup_dialog.show()

    # def scan_timeout(self):
    #     self.lx.showRealTimeImage()

    def view(self):
        self.lx.pcl()

    def __del__(self) :
        if rsData.r:
            rsData.r.close()
        self.stop()
        # self.scan_Timer.stop()

    def start(self):
        if not self.thread_streaming or not self.thread_streaming.is_alive():
            self.thread_streaming = threading.Thread(target=self.cam_bll.streaming)
            rsData.b_stop_streaming = False
            self.thread_streaming.start()
            self.lx.clearImageWidget()
            # self.scan_Timer.start()
            if self.cam_bll.initilized_cam:
                self.radio_review.setEnabled(False)
                self.modeCntl.reviewListWidget.setEnabled(False)
        else:
            # self.scan_Timer.stop()
            self.radio_review.setEnabled(True)
            self.lx.clearImageWidget()
            rsData.b_stop_streaming = True
            self.thread_streaming.join()
            self.thread_streaming = None
            self.modeCntl.reviewListWidget.setEnabled(True)

    def stop(self):
        rsData.b_stop_streaming = True
        # self.scan_Timer.stop()
        self.modeCntl.reviewListWidget.setEnabled(True)
        if self.thread_streaming and self.thread_streaming.is_alive():
           self.thread_streaming.join()

    def get_current_pose(self):
        if rsData.r:
            return rsData.r.get_cartesian()
        return []

    def closeAbb(self):
        try:
            if rsData.r:
                rsData.r.close()
                rsData.r = None
                # rsData.r.working = False
                # rsData.r.wait()
        except Exception as e:
            rsData.log.error(e)

    def openAbb(self):
        try:
            if rsData.r and len(rsData.r.get_cartesian())>0:
                self.closeAbb()
                self.getRobotInfo_timer.stop()
            else:
                robotIp = ConfBLL.readconf("robot","robotip")
                if check_ip(robotIp):
                    rsData.r = Robot(ip=robotIp)
                    self.getRobotInfo_timer.start()
                else:
                    message = QtWidgets.QMessageBox.critical(self, "vision", "connect robotip {} failed!".format(robotIp))

        except Exception as e:
            rsData.state[1]=EStateMode.eErr_Critical

    def capture(self):
        # if rsData.r:
        #     self.modeCntl.caliWidget.robotPos= rsData.r.get_cartesian()
        if rsData.mode == EMode.eCali :
            record = "{}\{}".format(os.path.abspath(self.save_dir), self.datafile)
            record_tcp_path = "{}\{}".format(os.path.abspath(self.save_dir), self.tcpfile)
            pose = rsData.robotPos
            xyz = pose[0]
            cart = pose[1]
            record_tcp = "{},{},{},{},{},{},{}".format(xyz[0], xyz[1], xyz[2], cart[0], cart[1], cart[2],cart[3])
            # ct = time.time()
            # local_time = time.localtime(ct)
            # timestamp = time.strftime("%Y%m%d_%H%M%S", local_time)

            curGoalIdx = self.caliWidget.cali.table.currentRow()
            colorFileName = "{}\{}".format(os.path.abspath(self.save_dir), "Color_{:0>2d}.png".format(curGoalIdx))
            depthFileName = "{}\{}".format(os.path.abspath(self.save_dir), "Depth_{:0>2d}.png".format(curGoalIdx))
            colorFileName = self.increment_path(colorFileName)
            depthFileName = self.increment_path(depthFileName)
            color_img = cv2.cvtColor(rsData.colorImg, cv2.COLOR_RGB2BGR)
            cv2.imwrite(colorFileName.as_posix(), color_img)
            # cv.imwrite(depthFileName.as_posix(),  kdata_lx.depthImg)
            # depth_mat = np.zeros((480, 640), np.uint16)
            #
            # for y in range(480):
            #     for x in range(640):
            #         depth_short = rsData.depthImg[y, x] * 10000
            #         depth_mat[y, x] = depth_short
            # cv2.imwrite(depthFileName.as_posix(), depth_mat)
            with open(record, 'a+') as f:
                f.write(colorFileName.as_posix() + '\n')
            with open(record_tcp_path, 'a+') as f:
                f.write(record_tcp + '\n')
        else:
            ct = time.time()
            local_time = time.localtime(ct)
            timestamp = time.strftime("%Y%m%d_%H%M%S", local_time)


            depth_img = rsData.depthImg * 10

                # depth_mat = np.zeros((480, 640), np.uint16)
                # for y in range(480):
                #     for x in range(640):
                #         depth_short = depth_img[y, x] * 10000
                #         depth_mat[y, x] = depth_short
                # depth_img = depth_mat

            file = QtCore.QFileInfo(rsData.cur_root_path)
            path = file.absoluteFilePath()
            cv2.imwrite("{}/frame-{}.depth.png".format(path,timestamp), depth_img)
            color_file_name = "{}/frame-{}.color.png".format(path,timestamp)
            rsData.colorImg = rsData.color_streaming
            cv2.imwrite(color_file_name, rsData.colorImg)
            # path = os.path.abspath(os.path.join(executable_path(), color_file_name))
            # self.store_file(path)
            # if self.write_pose:
            #     self.store_pose(self.get_current_pose())

            prefix = "frame-{}".format(timestamp)
            self.modeCntl.addItem(prefix)

            if self.view_after_capture:
                self.stop()
                self.modeCntl.LoadCurrentItem()
            else:
                try:
                    self.core.client_srv(prefix)
                except Exception as e:
                    message = QtWidgets.QMessageBox.critical(self, "python", "client_srv error!")

    def store_file(self,path):
        with open('file.txt', 'a+') as f:
            f.write(path + '\n')

    def store_pose(self,pose):
        with open('pose.txt', 'a+') as f:
            xyz = pose[0]
            qua = pose[1]
            p = "{},{},{},{},{},{},{}".format(xyz[0], xyz[1], xyz[2], qua[0], qua[1],qua[2],qua[3])
            f.write(p + '\n')

    def closeEvent(self, event):
        if rsData.r:
            rsData.r.close()
        self.stop()
        rsData.b_ws_server_stop = True
        self.t.join()


if __name__ == '__main__':
    # QtWebEngine.QtWebEngine.initialize()

    # engine = QtQml.QQmlApplicationEngine()
    # cx = engine.rootContext()
    # cx.setContextProperty("myUrl", "./3dProject/3dEntity/ur-ur5e.html")
    # cx.setContextProperty("isOffTheRecord", "")
    # cx.setContextProperty("viewStorageName", "")
    # engine.load(QtCore.QUrl("./robot.qml"))
    app = QtWidgets.QApplication(sys.argv)
    #
    #
    rs = RealSenseWidget()
    rs.show()
    #
    desktop = QtWidgets.QApplication.desktop()
    rect = desktop.geometry()
    pos = QtCore.QPoint(rect.x() + (rect.width() - rs.width()) / 2, rect.y() + (rect.height() - rs.height()) / 2)
    rs.move(pos)
    # rs.dock.setFocus()
    # Start the event loop.
    app.exec_()


    # objRB = Robot()
    # RB = GLWidget(None,objRB)
    # RB.show()
    # app.exec_()
    # gl = GLWidget()





 