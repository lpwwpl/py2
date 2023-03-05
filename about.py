import os

from PySide2 import QtGui, QtCore, QtWidgets

class About(QtWidgets.QDialog):

    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self, parent,
                               QtCore.Qt.Window | QtCore.Qt.WindowCloseButtonHint)

        self.setWindowTitle("About")

        mainLayout = QtWidgets.QVBoxLayout()
        mainLayout.setContentsMargins(0,0,0,0)
        self.setLayout(mainLayout)

        self.setFixedSize(500, 270)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0,0,0,0)
        form.addRow("<b>Version</b>", QtWidgets.QLabel("0.1.0"))
        form.addRow("<b>Author</b>", QtWidgets.QLabel("lpw"))
        form.addRow("<b>Email</b>", QtWidgets.QLabel("250531059@qq.com"))

        mainLayout.addLayout(form)

        hbox = QtWidgets.QHBoxLayout()
        hbox.setContentsMargins(5, 0, 5, 0)
        mainLayout.addLayout(hbox)

        self.label = QtWidgets.QLabel("External Libraries:")
        hbox.addWidget(self.label)

        hbox.addStretch(1)

        # licenseButton = QtWidgets.QPushButton("License")
        # licenseButton.setCheckable(True)
        # licenseButton.clicked.connect(self.showLicense)
        # hbox.addWidget(licenseButton)

        self.view = QtWidgets.QStackedWidget()
        mainLayout.addWidget(self.view)
        #
        # self.licenseEdit = QtWidgets.QTextEdit()
        # file = open(os.path.join("Resources", "LICENSE"), "r")
        # self.licenseEdit.setText(file.read())
        # file.close()
        #
        # self.view.addWidget(self.licenseEdit)

        self.hide()

    # def showLicense(self, checked):
    #     if checked:
    #         self.view.setCurrentIndex(1)
    #         self.label.hide()
    #     else:
    #         self.view.setCurrentIndex(0)
    #         self.label.show()
