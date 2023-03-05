from PySide2 import QtCore, QtGui, QtWidgets
import rsData
from UtilSet import *
LOCATION_INDEX=QtCore.Qt.UserRole + 1

class ReviewImageList(QtWidgets.QListWidget):
    def __init__(self, parent=None):
        QtWidgets.QListWidget.__init__(self, parent)
        self.menu = QtWidgets.QMenu()
        self.InitMenu()

    def contextMenuEvent(self, event) -> None:
        self.menu.exec_(event.globalPos())


    def InitMenu(self):
        self.action = QtWidgets.QAction("Delete")
        self.action.setData("Delete")
        self.menu.addAction(self.action)

class CaliParamActionWidget(QtWidgets.QWidget):
    def __init__(self,parent=None) :
        QtWidgets.QWidget.__init__(self, parent)
        self.initUI()

    def initUI(self):
        self.cali = CaliActionWidget()
        self.btn_goto=QtWidgets.QPushButton("Goto")
        self.btn_goto.clicked.connect(self.startNavi)
        mainLayout = QtWidgets.QGridLayout()
        mainLayout.setContentsMargins(0,0,0,0)
        mainLayout.addWidget(self.cali,0,0,8,5)
        mainLayout.addWidget(self.btn_goto,8,4,1,1)
        self.setLayout(mainLayout)

    def startNavi(self):
        self.cali.startNavi()

    def reset(self,path):
        self.cali.reset(path)

class CaliActionWidget(QtWidgets.QWidget):
    def __init__(self,parent=None) :
        QtWidgets.QWidget.__init__(self, parent)
        self.initUI()

    def initUI(self):
        self.table = QtWidgets.QTableWidget()
        self.table.setRowCount(0)
        self.table.setColumnCount(6)  #
        self.table.setSortingEnabled(True)
        self.horizontalHeader = ["Joints0", "Joints1", "Joints2", "Joints3", "Joints4", "Joints5"]
        self.table.setHorizontalHeaderLabels(self.horizontalHeader)
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self.table.setSelectionMode(QtWidgets.QTableWidget.SingleSelection)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(5, QtWidgets.QHeaderView.Stretch)
        # self.btn_goto=QtWidgets.QPushButton("Goto")
        # self.btn_cali_stop=QtWidgets.QPushButton("Stop")
        # self.btn_write_back = QtWidgets.QPushButton("WriteBack")
        # self.btn_export = QtWidgets.QPushButton("export")
        # self.gridlayout=QtWidgets.QGridLayout()
        # self.gridlayout.addWidget(self.btn_goto,0,1,1,1)
        # self.gridlayout.addWidget(self.btn_cali_stop,0,1,1,1)
        # self.gridlayout.addWidget(self.btn_write_back,0,2,1,1)
        # self.gridlayout.addWidget(self.btn_export,0,3,1,1,)
        # self.btn_goto.clicked.connect(self.startNavi)
        # self.btn_write_back.clicked.connect(self.write_back)
        # self.btn_export.clicked.connect(self.export)
        # self.btn_cali_stop.clicked.connect(self.cali_stop)
        # self.btn_widget = QtWidgets.QWidget()
        # self.btn_widget.setLayout(self.gridlayout)
        mainLayout = QtWidgets.QVBoxLayout()
        mainLayout.setContentsMargins(0,0,0,0)
        mainLayout.addWidget(self.table)
        # mainLayout.addWidget(self.btn_widget)
        self.setLayout(mainLayout)
        # self.curGoalIdx = 0
        # self.table.setMinimumHeight(480)
        self.setWindowFlag( QtCore.Qt.WindowStaysOnTopHint)
        self.curItem = None

    def reset(self,path):
        self.table.blockSignals(True)
        self.table.clear()
        # rsData.r = None
        tPath = "{}/{}".format(path,"joints.txt")
        fileInfo = QtCore.QFileInfo(tPath)
        if not fileInfo.exists():
            return
        with open(tPath) as f:
            lines = f.readlines()
            self.table.setRowCount(len(lines))
            for i in range(len(lines)):
                joints = lines[i].split(',')
                if len(joints) == 6:
                    newItem1 = QtWidgets.QTableWidgetItem(str(joints[0]))
                    #newItem1.setFlags(Qt.ItemIsEnabled | Qt.ItemIsEditable)
                    newItem1.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                    self.table.setItem(i, 0, newItem1)

                    newItem1 = QtWidgets.QTableWidgetItem(joints[1])
                    newItem1.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                    self.table.setItem(i,1, newItem1)

                    newItem1 = QtWidgets.QTableWidgetItem(joints[2])
                    newItem1.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                    self.table.setItem(i,2, newItem1)

                    newItem1 = QtWidgets.QTableWidgetItem(joints[3])
                    newItem1.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                    self.table.setItem(i,3, newItem1)

                    newItem1 = QtWidgets.QTableWidgetItem(joints[4])
                    newItem1.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                    self.table.setItem(i,4, newItem1)

                    newItem1 = QtWidgets.QTableWidgetItem(str(joints[5]))
                    newItem1.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
                    self.table.setItem(i,5, newItem1)
            if self.table.rowCount()>0:
                self.table.setCurrentItem(self.table.item(0,0))
                self.table.selectRow(0)

        self.table.doubleClicked.connect(self.doubleClick)
        self.table.blockSignals(False)


    def doubleClick(self, index):
        qindex = (QtCore.QModelIndex)(index)
        column = qindex.column()

        item = self.table.item(qindex.row(), qindex.column())
        self.table.openPersistentEditor(item)
        self.table.editItem(item)
        self.curItem = item
        self.editWidget = self.table.indexWidget(index)
        self.editWidget.setFocus()
        self.editWidget.editingFinished.connect(self.closeEditing)
        return

    def closeEditing(self):
        if self.curItem:
            self.editWidget.close()
            self.table.closePersistentEditor(self.curItem)
            # self.table.item(self.curItem.row(), 4).setText("")
            self.curItem = None

    def cali_stop(self):
        pass

    def completeNavi(self):
        curGoalIdx = self.table.currentRow()
        color = QtGui.QColor(255, 69, 0)
        self.table.item(curGoalIdx, 0).setBackground(color)
        self.table.item(curGoalIdx, 1).setBackground(color)
        self.table.item(curGoalIdx, 2).setBackground(color)
        self.table.item(curGoalIdx, 3).setBackground(color)
        self.table.item(curGoalIdx, 4).setBackground(color)
        self.table.item(curGoalIdx, 5).setBackground(color)

    def startNavi(self):

        curItem = self.table.currentItem()
        if curItem.row() ==0:
            rsData.need_new_folder = True
        color = QtGui.QColor(255,69,0)
        curGoalIdx = self.table.currentRow()
        # self.table.item(self.curGoalIdx,0).setBackground(color)
        # self.table.item(self.curGoalIdx,1).setBackground(color)
        # self.table.item(self.curGoalIdx, 2).setBackground(color)
        # self.table.item(self.curGoalIdx, 3).setBackground(color)
        # self.table.item(self.curGoalIdx, 4).setBackground(color)
        # self.table.item(self.curGoalIdx, 5).setBackground(color)

        joints0=float(self.table.item(curGoalIdx,0).text())
        joints1=float(self.table.item(curGoalIdx,1).text())
        joints2=float(self.table.item(curGoalIdx,2).text())
        joints3=float(self.table.item(curGoalIdx,3).text())
        joints4=float(self.table.item(curGoalIdx,4).text())
        joints5=float(self.table.item(curGoalIdx,5).text())
        if rsData.r:
            joints = [joints0,joints1,joints2,joints3,joints4,joints5]
            rsData.r.set_joints(joints)
            print("set_joints")
            # rsData.curJoints = joints
        curItem = self.table.currentItem()
        row = curItem.row()+1
        if row>=self.table.rowCount():
            row = 0

        self.table.setCurrentItem(self.table.item(row,0))
        self.table.selectRow(row)
        self.table.setFocus()
        # self.curGoalIdx = row

    def write_back(self):
        curGoalIdx = self.table.currentRow()
        if rsData.r and self.curGoalIdx >= 0:
            joints=rsData.r.get_joints()
            self.table.item(curGoalIdx, 0).setText(str(joints[0]))
            self.table.item(curGoalIdx, 1).setText(str(joints[1]))
            self.table.item(curGoalIdx, 2).setText(str(joints[2]))
            self.table.item(curGoalIdx, 3).setText(str(joints[3]))
            self.table.item(curGoalIdx, 4).setText(str(joints[4]))
            self.table.item(curGoalIdx, 5).setText(str(joints[5]))

    def export(self):
        with open("data/runs/calibration_collect/exp/joints.txt", 'w+') as f:
            for i in range(self.table.rowCount()):
                joints0 = self.table.item(i, 0)
                joints1 = self.table.item(i, 1)
                joints2 = self.table.item(i, 2)
                joints3 = self.table.item(i, 3)
                joints4 = self.table.item(i, 4)
                joints5 = self.table.item(i, 5)
                str = "{},{},{},{},{},{}\n".format(joints0,joints1,joints2,joints3,joints4,joints5)
                f.write(str)


class ModeControlTabWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.initUI()
        self.core = None

    def setCore(self,core):
        self.core = core

    def initUI(self):
        self.reviewListWidget = ReviewImageList()
        self.reviewListWidget.setIconSize(QtCore.QSize(118, 89))
        self.reviewListWidget.setSpacing(0)
        self.reviewListWidget.setWordWrap(True)
        self.reviewListWidget.setContentsMargins(0,0,0,0)
        self.reviewListWidget.setTextElideMode(QtCore.Qt.ElideRight)
        self.reviewListWidget.itemDoubleClicked.connect(self.OnItemDoubleClicked)
        self.reviewListWidget.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.reviewListWidget.setViewMode(QtWidgets.QListWidget.IconMode)
        self.reviewListWidget.setFixedWidth(300)
        self.reviewListWidget.setMovement(QtWidgets.QListView.Static)
        self.reviewListWidget.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.reviewListWidget.menu.triggered.connect(self.reviewList_menu_trigger)

        # self.caliWidget = CaliParamActionWidget()

        self.tab = QtWidgets.QTabWidget()
        self.tab.setFixedWidth(300)
        self.tab.addTab(self.reviewListWidget,"Review")
        # self.tab.addTab(self.caliWidget, "Calibration")
        self.tab.setContentsMargins(0,0,0,0)
        self.layout = QtWidgets.QVBoxLayout()
        # self.layout.addWidget(self.reviewListWidget)
        self.layout.addWidget(self.tab)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)

    def reviewList_menu_trigger(self,action):
        actiondata = action.data()
        if actiondata == "Delete":
            cur_item = self.reviewListWidget.currentItem()
            prefix = cur_item.data(LOCATION_INDEX)
            colorPath = "{}/{}.color.png".format(rsData.cur_root_path,prefix)
            depthPath= "{}/{}.depth.png".format(rsData.cur_root_path,prefix)
            inferPath = "{}/{}.infer.png".format(rsData.cur_root_path,prefix)
            if os.path.exists(colorPath):
                os.remove(colorPath)
                row = self.reviewListWidget.row(cur_item)
                self.reviewListWidget.takeItem(row)
                self.reviewListWidget.update()
            if os.path.exists(depthPath):
                os.remove(depthPath)
            if os.path.exists(inferPath):
                os.remove(inferPath)

    def addItem(self,prefix):
        imageIcon = QtGui.QIcon()
        compName = "{}/{}.color.png".format(rsData.cur_root_path,prefix)
        image = QtGui.QImage(compName)
        scaleImage = image.scaled(118, 89)
        imageIcon.addPixmap(QtGui.QPixmap.fromImage(scaleImage))

        item = QtWidgets.QListWidgetItem(self.reviewListWidget)
        item.setData(QtCore.Qt.DisplayRole, prefix)
        item.setData(LOCATION_INDEX, prefix)
        item.setIcon(imageIcon)
        item.setFont(QtGui.QFont("宋体", 9, QtGui.QFont.Bold))
        item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
        item.setSizeHint(QtCore.QSize(130, 100))
        item.setToolTip(prefix)
        self.reviewListWidget.scrollToItem(item)
        self.reviewListWidget.setCurrentItem(item)

    def LoadCurrentItem(self):
        item = self.reviewListWidget.currentItem()
        self.LoadItem(item)

    def LoadItem(self,item=None):
        if item:
            self.OnItemDoubleClicked(item)

    def LoadPiclx(self,files_path):
        self.reviewListWidget.clear()

        imageIcon = QtGui.QIcon()
        scaleImage = None

        # files_path = "./picslx"
        strList = []
        for fl in os.listdir(files_path):
            fileInfo = QtCore.QFileInfo(fl)
            suffix = fileInfo.completeSuffix()
            baseName = fileInfo.baseName()
            depthPath = "{}/{}.depth.png".format(files_path,baseName)
            if suffix.endswith('color.png') :#and os.path.exists(depthPath):
                strList.append(fileInfo.baseName())

        strList.sort()
        for name in strList:
            compName = "{}/{}.color.png".format(files_path,name)
            image = QtGui.QImage(compName)
            scaleImage = image.scaled(118, 89)
            imageIcon.addPixmap(QtGui.QPixmap.fromImage(scaleImage))

            item = QtWidgets.QListWidgetItem(self.reviewListWidget)
            item.setData(QtCore.Qt.DisplayRole, name)
            item.setData(LOCATION_INDEX,name)
            item.setIcon(imageIcon)
            item.setFont(QtGui.QFont("宋体",9,QtGui.QFont.Bold))
            item.setTextAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            item.setSizeHint(QtCore.QSize(130, 100))
            item.setToolTip(name)

    def OnItemDoubleClicked(self,item):
        try:
            filePath = item.data(LOCATION_INDEX)
            self.core.client_srv(filePath)
        except Exception as e:
            rsData.log.error(e)
            message = QtWidgets.QMessageBox.critical(self, "python", "client_srv error!")