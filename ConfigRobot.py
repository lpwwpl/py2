from GlobalFunc import *
import numpy as np

class ConfigRobot(object):
	"""docstring for ConfigRobot"""
	def __init__(self):
		super(ConfigRobot, self).__init__()
		self.d = np.array([0, 290, 270, 70,   0, 0 ])
		self.a = np.array([0,   0,   0,  0, 302, 72])
		self.alpha = np.array([0,  0,  0,  0,  0,  0])
		self.q_init = np.array([0, 42,   -31,  11,   -86,  40, -179])


		# self.q_init = np.array([30, 30,   30,  30,   30,  30, 30])
	def get_q_init(self):
		return self.q_init