from PySide2 import QtQuick,QtQml,QtWidgets,QtCore,QtGui
from PySide2.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
from PySide2.QtWebChannel import QWebChannel
from PySide2.QtWebSockets import QWebSocketServer
from PySide2.QtWebChannel import QWebChannelAbstractTransport
from PySide2.QtNetwork import QHostAddress
import rsData
from confBLL import ConfBLL
import time
class WebSocketTransport(QWebChannelAbstractTransport):
    def __init__(self,socket):
        QWebChannelAbstractTransport.__init__(self)
        self.socket = socket
        socket.textMessageReceived.connect(self.textMessageReceived)
        socket.disconnected.connect(self.deleteLater())

    def sendMessage(self,msg):
        json = QtCore.QJsonDocument(msg)
        self.socket.sendTextMessage(json.toJson(QtCore.QJsonDocument.Compact))

    def textMessageReceived(self,msg):
        error = QtCore.QJsonDocument.fromJson(msg.toUtf8())
        if error.error:
            return
        elif not msg.isObject():
            return
        self.messageReceived.emit(msg.object())


class WebSocketClientWrapper(QtCore.QObject):
    clientConnected = QtCore.Signal(WebSocketTransport)
    clientDisconnected = QtCore.Signal()
    def __init__(self,server):
        QtCore.QObject.__init__(self)
        self.server = server
        self.server.newConnection.connect(self.handleNewConnection)

    def handleNewConnection(self):
        socket = self.server.nextPendingConnection()
        if socket:
            socket.disconnected.connect(self.clientDisconnected)
            self.clientConnected.emit(WebSocketTransport(socket))

class BridgetData(QtCore.QObject):
    jointsChanged = QtCore.Signal(str)
    def __init__(self):
        QtCore.QObject.__init__(self)

    @QtCore.Slot()
    def getJoints(self):
        return rsData.curJoints

class ClickView(QtQuick.QQuickView):
    def __init__(self):
        QtQuick.QQuickView.__init__(self)
        # self.engine().addImportPath()
        self.setResizeMode(QtQuick.QQuickView.SizeRootObjectToView)
        self.setFlags(QtCore.Qt.FramelessWindowHint|QtCore.Qt.Window|QtCore.Qt.WindowStaysOnTopHint)
        self.initQWebChannel()

    def __del__(self):
        if self.clientWrapper:
            self.clientWrapper.deleteLater()
        if self.m_server:
            if self.m_server.isListening():
                self.m_server.close()
            self.m_server.deleteLater()

    def initQWebChannel(self):
        # self.m_server = QWebSocketServer("SocketServer",QWebSocketServer.NonSecureMode,self)
        # connected = False
        # port = 12345
        #
        # while True:
        #     connected = self.m_server.listen(QHostAddress.LocalHost,port)
        #     if connected:
        #         break
        #     time.sleep(2)

        self.m_server.setMaxPendingConnections(1)
        self.data = BridgetData()
        self.m_channel = QWebChannel()
        self.m_channel.registerObject("any_robot",self.data)

        # self.clientWrapper = WebSocketClientWrapper(self.m_server)
        # self.clientWrapper.clientConnected.connect(self.m_channel.connectTo)
        # self.clientWrapper.clientConnected.connect(self.onConnected)
        # self.clientWrapper.clientDisconnected.connect(self.onDisconnected)

    def onConnected(self):
        pass

    def onDisconnected(self):
        pass
    # def mousePressEvent(self, event):
    #     pass

class MeWebView(QWebEngineView):
    def __init__(self):
        QWebEngineView.__init__(self)
        # self.engine().addImportPath()
        self.initQWebChannel()

    def __del__(self):
        pass

    def initQWebChannel(self):

        rsData.url = ConfBLL.readconf("param","url")
        self.load(rsData.url)
        # self.m_server = QWebSocketServer("SocketServer",QWebSocketServer.NonSecureMode,self)
        # connected = False
        # port = 12345
        #
        # while True:
        #     connected = self.m_server.listen(QHostAddress.LocalHost,port)
        #     if connected:
        #         break
        #     time.sleep(2)
        #
        # self.m_server.setMaxPendingConnections(1)
        # data = BridgetData()
        # self.m_channel = QWebChannel()
        # self.m_channel.registerObject("any_robot",data)
        # # self.page().setWebChannel(m_channel)
        #
        # self.clientWrapper = WebSocketClientWrapper(self.m_server)
        # ret = self.clientWrapper.clientConnected.connect(self.m_channel.connectTo)
        # self.clientWrapper.clientConnected.connect(self.onConnected)
        # self.clientWrapper.clientDisconnected.connect(self.onDisconnected)

    def onConnected(self):
        pass

    def onDisconnected(self):
        pass

class RobotQmlWidget(QtWidgets.QWidget):
    def __init__(self,parent=None):
        QtWidgets.QWidget.__init__(self,parent)
        # self.engine = engine
        self.initUi()


    def initUi(self):
        # self.qml = ClickView()
        # ctx = self.qml.rootContext()
        # ctx.setContextProperty("myUrl", "http://localhost:8080/3dProject/3dEntity/ur-ur5e.html")  # ur-ur5e.html
        # self.qml.setSource(QtCore.QUrl("./robot.qml"))
        # self.qmlWidget = QtWidgets.QWidget.createWindowContainer(self.qml)
        #
        # self.layout = QtWidgets.QHBoxLayout()
        # self.layout.setContentsMargins(0, 0, 0, 0)
        # self.layout.addWidget(self.qmlWidget)
        # self.setLayout(self.layout)

        self.webView = MeWebView()


        self.layout = QtWidgets.QHBoxLayout()
        self.layout.setContentsMargins(0,0,0,0)
        self.layout.addWidget(self.webView)
        self.setLayout(self.layout)
