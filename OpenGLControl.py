from PySide2 import QtCore, QtGui
from PySide2 import QtOpenGL
from OpenGL import GLU
from OpenGL.GL import *
from OpenGL.GLUT import *
from numpy import array, arange
from STLFile import *
# from ConfigRobot import *
from GlobalFunc import *
import math

from ConfigRobot import ConfigRobot
# from kinematics import *


class Robot(object):
	"""docstring for Robot"""
	def __init__(self):
		super(Robot, self).__init__()
		self.cf = ConfigRobot()
		self.q = self.cf.q_init
		self.d = self.cf.d
		self.a = self.cf.a
		self.alpha = self.cf.alpha
		self.JVars = self.cf.q_init[1:]
		# joint_transforms, bTee = forward_kinematics(Tt)
		# print(bTee)

class GLWidget(QtOpenGL.QGLWidget):
	xRotationChanged = QtCore.Signal(int)
	yRotationChanged = QtCore.Signal(int)
	zRotationChanged = QtCore.Signal(int)

	def __init__(self, parent=None, objRobot=None):
		super(GLWidget, self).__init__(parent)
		self.objRobot = objRobot
		self.xRot = -2584
		self.yRot = 1376
		self.zRot = 0.0
		self.z_zoom = -3500#00 #-3500
		self.xTran = 0
		self.yTran = 0
		self.isDrawGrid = True;
		print("Loading stl files...")
		self.model0 = loader('STLFile/base_link.stl')

		self.model1 = loader('STLFile/link_1.stl')
		self.model2 = loader('STLFile/link_2.stl')
		self.model3 = loader('STLFile/link_3.stl')
		self.model4 = loader('STLFile/link_4.stl')
		self.model5 = loader('STLFile/link_5.stl')
		self.model6 = loader('STLFile/link_6.stl')
		# self.model0 = loader('STLFile/Link0.stl')
		# self.model1 = loader('STLFile/Link1.STL')
		# self.model2 = loader('STLFile/Link2.STL')
		# self.model3 = loader('STLFile/Link3.STL')
		# self.model4 = loader('STLFile/Link4.STL')
		# self.model5 = loader('STLFile/tool.STL')
		# self.model6 = loader('STLFile/link_6.STL')
		print("All done.")

		self.listPoints = np.array([[0,0,0]])
		self.AllList = np.array([self.listPoints])
		self.stt = np.array([])
		self.color=np.array([0])

	def setXRotation(self, angle):
		self.normalizeAngle(angle)
		if angle != self.xRot:
			self.xRot = angle
			self.xRotationChanged.emit(angle)
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
			self.updateGL()

	def setYRotation(self, angle):
		self.normalizeAngle(angle)
		if angle != self.yRot:
			self.yRot = angle
			self.yRotationChanged.emit(angle)
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
			# self.updateGL()

	def setZRotation(self, angle):
		self.normalizeAngle(angle)
		if angle != self.zRot:
			self.zRot = angle
			self.zRotationChanged.emit(angle)
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
			self.updateGL()

	def setXYTranslate(self, dx, dy):
		self.xTran += 3.0 * dx
		self.yTran -= 3.0 * dy
		self.updateGL()

	def setZoom(self, zoom):
		self.z_zoom = zoom
		self.updateGL()

	def updateJoint(self):
		self.updateGL()

	def initializeGL(self):
		lightPos = (5.0, 5.0, 10.0, 1.0)
		reflectance1 = (0.8, 0.1, 0.0, 1.0)
		reflectance2 = (0.0, 0.8, 0.2, 1.0)
		reflectance3 = (0.2, 0.2, 1.0, 1.0)

		ambientLight = [0.7, 0.7, 0.7, 1.0]
		diffuseLight = [0.7, 0.8, 0.8, 1.0]
		specularLight = [0.4, 0.4, 0.4, 1.0]
		positionLight = [20.0, 20.0, 20.0, 0.0]

		glLightfv(GL_LIGHT0, GL_AMBIENT, ambientLight);
		glLightfv(GL_LIGHT0, GL_DIFFUSE, diffuseLight)
		glLightfv(GL_LIGHT0, GL_SPECULAR, specularLight)
		glLightModelf(GL_LIGHT_MODEL_TWO_SIDE, 1.0)
		glLightfv(GL_LIGHT0, GL_POSITION, positionLight)

		glEnable(GL_LIGHTING)
		glEnable(GL_LIGHT0)
		glEnable(GL_DEPTH_TEST)
		glEnable(GL_NORMALIZE)
		# glEnable(GL_BLEND);
		glClearColor(0.0, 0.0, 0.0, 1.0)


	def drawGL(self):

		# self.d = np.array([ 0, 290, 0, 0, -302, -72, 0])
		# self.a = np.array([ 0, 0, 270, 70, 0, 0, 0])
		# self.alpha = DegToRad(np.array([0,-90, 0, 90, -90, 90, 180]))
		# self.q_init = DegToRad(np.array([0,0, -90, 0, 0, 0, 180]))

		glPushMatrix()
		if self.isDrawGrid:
			self.drawGrid()
		# Base
		self.setupColor([96.0 / 255, 96 / 255.0, 192.0 / 255])
		glScalef(1000,1000,1000)
		self.model0.draw()
		self.setupColor([169.0 / 255, 169.0 / 255, 169.0 / 255])
		glScalef(0.001, 0.001, 0.001)

		# Link1
		glTranslatef(0.0, 0.0, self.objRobot.d[0])
		glRotatef(self.objRobot.JVars[0], 0.0, 0.0, 1.0)
		glTranslatef(self.objRobot.a[0], 0.0, 0.0)
		glRotatef(self.objRobot.alpha[0], 1.0, 0.0, 0.0)
		glScalef(1000, 1000, 1000)
		self.model1.draw()
		glScalef(0.001, 0.001, 0.001)


		#Link2
		self.setupColor([90.0 / 255, 150.0 / 255, 9.0 / 255])
		glTranslatef(0.0, 0.0, self.objRobot.d[1])
		glRotatef((self.objRobot.JVars[1]), 0.0, 1.0,0.0)
		glTranslatef(self.objRobot.a[1], 0.0, 0.0)
		glRotatef((self.objRobot.alpha[1]), 1.0, 0.0, 0.0)
		glScalef(1000, 1000, 1000)
		self.model2.draw()
		glScalef(0.001, 0.001, 0.001)

		#Link3
		self.setupColor([255.0 / 255, 255.0 / 255, 9.0 / 255])
		glTranslatef(0.0, 0.0, self.objRobot.d[2])
		glRotatef((self.objRobot.JVars[2]), 0.0, 1.0, 0.0)
		glTranslatef(self.objRobot.a[2], 0.0, 0.0)
		glRotatef((self.objRobot.alpha[2]), 1.0, 0.0, 0.0)
		glScalef(1000, 1000, 1000)
		self.model3.draw()
		glScalef(0.001, 0.001, 0.001)

		#Link4
		self.setupColor([105.0 / 255, 180.0 / 255, 0 / 255])
		glTranslatef(0.0, 0.0, self.objRobot.d[3])
		glRotatef((self.objRobot.JVars[3]), 1.0, 0.0, 0.0)
		glTranslatef(self.objRobot.a[3], 0.0, 0.0)
		glRotatef((self.objRobot.alpha[3]), 1.0, 0.0, 0.0)
		glScalef(1000, 1000, 1000)
		self.model4.draw()
		glScalef(0.001, 0.001, 0.001)

		# Link5
		self.setupColor([105.0 / 255, 180.0 / 255, 0 / 255])
		glTranslatef(0.0, 0.0, self.objRobot.d[4])

		glTranslatef(self.objRobot.a[4], 0.0, 0.0)
		glRotatef((self.objRobot.JVars[4]), 0.0, 1.0, 0.0)
		glRotatef((self.objRobot.alpha[4]), 1.0, 0.0, 0.0)
		glScalef(1000, 1000, 1000)
		self.model5.draw()
		glScalef(0.001, 0.001, 0.001)

		# Link6
		self.setupColor([0.0/255, 180.0/255, 84.0/255])
		glTranslatef(0.0, 0.0, self.objRobot.d[5])
		glRotatef((self.objRobot.JVars[5]), 1.0, 0.0, 0.0)
		glTranslatef(self.objRobot.a[5], 0.0, 0.0)
		glRotatef((self.objRobot.alpha[5]), 1.0, 0.0, 0.0)
		glScalef(1000, 1000, 1000)
		self.model6.draw()
		glScalef(0.001, 0.001, 0.001)
		glPopMatrix()

	def paintGL(self):
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
		glPushMatrix()
		glTranslate(0, 0, self.z_zoom)
		glTranslate(self.xTran, self.yTran, 0)
		glRotated(self.xRot/16.0, 1.0, 0.0, 0.0)
		glRotated(self.yRot/16.0, 0.0, 1.0, 0.0)
		glRotated(self.zRot/16.0, 0.0, 0.0, 1.0)
		glRotated(+90.0, 1.0, 0.0, 0.0)
		self.drawGL()
		self.DrawPoint([255.0/255, 255.0/255, 255.0/255.0], 1.5)
		glPopMatrix()

	def DrawPoint(self, color, size):
		glPushMatrix()
		glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, self.color);
		glPointSize(size);		
		for i in np.arange(len(self.listPoints)-1):
			if self.color[i] == 1:
				glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, [1.0, 1.0, 1.0]);
				glBegin(GL_LINES);
				glVertex3f(self.listPoints[i][0], self.listPoints[i][1], self.listPoints[i][2])
				glVertex3f(self.listPoints[i+1][0], self.listPoints[i+1][1], self.listPoints[i+1][2])
				glEnd()
		glPopMatrix()

	def resizeGL(self, width, height):
		side = min(width, height)
		if side < 0:
			return
		glViewport(0, 0, width, height)
		glMatrixMode(GL_PROJECTION)
		glLoadIdentity()
		GLU.gluPerspective(35.0, width / float(height), 1.0, 20000.0)
		glMatrixMode(GL_MODELVIEW)
		glLoadIdentity()
		glTranslated(0.0, 0.0, -40.0)

	def mousePressEvent(self, event):
		self.lastPos = event.pos()

	def drawGrid(self):
		glPushMatrix()
		# color = [255.0/255, 57.0/255, 0.0/255]
		color = [8.0/255, 108.0/255, 162.0/255]
		glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);
		step = 50
		num = 15
		for i in arange(-num, num+1):
			glBegin(GL_LINES)
			glVertex3f(i*step, -num * step, 0)
			glVertex3f(i*step, num*step, 0)
			glVertex3f(-num * step, i*step, 0)
			glVertex3f(num*step, i*step, 0)
			glEnd()
		glPopMatrix()

	def mouseMoveEvent(self, event):
		dx = event.x() - self.lastPos.x()
		dy = event.y() - self.lastPos.y()
		if event.buttons() & QtCore.Qt.LeftButton:
			self.setXRotation(self.xRot + 4 * dy)
			self.setYRotation(self.yRot - 4 * dx)
		elif event.buttons() & QtCore.Qt.RightButton:
			self.setZoom(self.z_zoom + 5.0*dy)
		elif event.buttons() & QtCore.Qt.MidButton:
			self.setXYTranslate(dx, dy)
		self.lastPos = event.pos()


	def setupColor(self, color):
		glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, color);

	def xRotation(self):
		return self.xRot

	def yRotation(self):
		return self.yRot

	def zRotation(self):
		return self.zRot  

	def normalizeAngle(self, angle):
		while (angle < 0):
			angle += 360 * 16
		while (angle > 360 * 16):
			angle -= 360 * 16