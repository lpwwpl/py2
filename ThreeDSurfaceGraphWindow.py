import os
from PySide2 import QtWidgets,QtCore
os.environ['QT_API'] = 'pyside'
os.environ["FORCE_CPU"] = 'True'
from matplotlib.pyplot import figure
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import rsData

class ThreeDSurfaceGraphWindowDlg(QtWidgets.QDialog):
    def __init__(self):
        QtWidgets.QDialog.__init__(self)
        self.initUi()
        self.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                           QtWidgets.QSizePolicy.Preferred)
        self.setWindowFlag(QtCore.Qt.WindowStaysOnTopHint)


    def initUi(self):
        self.plt1= ThreeDSurfaceGraphWindow()
        self.plt2 = ThreeDSurfaceGraphWindow()
        self.layout = QtWidgets.QHBoxLayout()
        self.layout.addWidget(self.plt1)
        self.layout.addWidget(self.plt2)
        self.setLayout(self.layout)

class ThreeDSurfaceGraphWindow(FigureCanvas):  # Class for 3D window
    def __init__(self):
        self.plot_colorbar = None
        self.plot_figure = figure()
        FigureCanvas.__init__(self, self.plot_figure)
        self.axes = self.plot_figure.gca(projection='3d')
        self.setWindowTitle("")  # sets Window title

    def draw_graph(self, x, y, z, color, acolor):  # Function for graph plotting
        self.axes.clear()
        if self.plot_colorbar is not None:  # avoids adding one more colorbar at each draw operation
            self.plot_colorbar.remove()
        # plots the 3D surface plot
        plot_stuff = self.axes.scatter(x, y, z, c="tab:blue", linewidth=0, antialiased=True)

        for i in range(rsData.axisPoints.shape[0]):
            self.axes.scatter(rsData.axisPoints[i][:, 0], rsData.axisPoints[i][:, 1], rsData.axisPoints[i][:, 2],
                              c="tab:orange")

        self.axes.tick_params(axis='x', colors='red')
        self.axes.tick_params(axis='y', colors='green')
        self.axes.tick_params(axis='z', colors='blue')

        self.axes.zaxis.set_major_locator(LinearLocator(10))
        self.axes.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        # Add a color bar which maps values to colors.
        self.plot_colorbar = self.plot_figure.colorbar(plot_stuff, shrink=0.5, aspect=5)
        # draw plot
        self.draw()