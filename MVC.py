from PySide2 import QtWidgets,QtCore,QtGui

class PropertySheet(QtWidgets.QWidget):
    pass


class AbstractEntity(QtCore.QObject):
    def __init__(self, parent=None):
        QtCore.QObject.__init__(self, parent)

    def translate(self, delta):
        pass

    def rotate(self, rotation):
        pass

    def scale(self, scaling):
        pass

    def visible(self):
        pass

    def selected(self):
        pass

    def position(self):
        pass


class Model(AbstractEntity):
    def __init__(self, parent=None):
        AbstractEntity.__init__(self,parent)
        self.childModels = list


    def addChildModel(self,model):
        self.childModels.append(model)

    def removeChildModel(self,model):
        pass

class BaseTreeItem(QtWidgets.QTreeWidgetItem):
    def __init__(self, parent=None):
        QtWidgets.QTreeWidget.__init__(self,parent)


class CameraItem(BaseTreeItem):
    def __init__(self,parent=None):
        BaseTreeItem.__init__(self,parent)



class SceneTree(QtWidgets.QTreeWidget):
    def __init__(self, parent=None):
        QtWidgets.QTreeWidget.__init__(self, parent)

    def modelAdded(self, model):
        pass