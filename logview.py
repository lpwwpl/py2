from PySide2 import QtWidgets,QtCore,QtGui
import logging
from rsData import log
import os

class LogEmitter(QtCore.QObject):
    sigLog = QtCore.Signal(str)

class LogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        formatter = logging.Formatter("[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s",
                                      "%Y-%m-%d %H:%M:%S")
        self.setFormatter(formatter)
        self.setLevel(logging.DEBUG)
        self.emitter = LogEmitter()
    def emit(self, record):
        msg = self.format(record)
        self.emitter.sigLog.emit(msg)



class LogView(QtWidgets.QWidget):
    def __init__(self,parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.initUi()
        logHandler = LogHandler()
        log.addHandler(logHandler)
        log.setLevel(logging.INFO)
        logHandler.emitter.sigLog.connect(self.output)



    def initUi(self):
        self.textEdit = QtWidgets.QTextEdit()
        self.textEdit.setReadOnly(True)
        self.cmb_info_type = QtWidgets.QComboBox()
        self.cmb_info_type.blockSignals(True)
        self.cmb_info_type.addItem("INFO")
        self.cmb_info_type.addItem("DEBUG")
        self.cmb_info_type.addItem("WARNING")
        self.cmb_info_type.addItem("ERROR")
        self.cmb_info_type.addItem("CRITICAL")
        self.cmb_info_type.setCurrentText("INFO")
        self.cmb_info_type.blockSignals(False)
        self.cmb_info_type.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,QtWidgets.QSizePolicy.Policy.Preferred)
        self.cmb_info_type.currentTextChanged.connect(self.cmb_change)
        self.hLayout=QtWidgets.QHBoxLayout()
        self.hLayout.addWidget(self.cmb_info_type)
        self.hLayout.setContentsMargins(0,0,0,0)
        h1 = QtWidgets.QSpacerItem(40,20,QtWidgets.QSizePolicy.Policy.Expanding)

        self.hLayout.addSpacerItem(h1)
        self.cmb_widget = QtWidgets.QWidget()
        self.cmb_widget.setLayout(self.hLayout)
        self.layout = QtWidgets.QGridLayout()
        self.layout.addWidget(self.cmb_widget,0,0,1,1)
        # self.layout.setColumnStretch(1,1)
        self.layout.addWidget(self.textEdit,1,0,1,1)

        self.layout.setContentsMargins(0,0,0,0)
        self.setLayout(self.layout)
        self.textEdit.document().setMaximumBlockCount(100)
        self.textEdit.setVerticalScrollBarPolicy(QtGui.Qt.ScrollBarAsNeeded)
        # self.setMinimumWidth(480)
        # self.setFixedSize(QtCore.QSize(1280,960))
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)

    def cmb_change(self):
        if self.cmb_info_type.currentText() == "DEBUG":
            log.setLevel(logging.DEBUG)
        elif self.cmb_info_type.currentText() == "INFO":
            log.setLevel(logging.INFO)
        elif self.cmb_info_type.currentText() == "WARNING":
            log.setLevel(logging.WARNING)
        elif self.cmb_info_type.currentText() == "ERROR":
            log.setLevel(logging.ERROR)
        elif self.cmb_info_type.currentText() == "CRITICAL":
            log.setLevel(logging.CRITICAL)
    def output(self,line):
        self.textEdit.append(line)
