from math import *
import numpy as np

def RadToDeg(value):
	return value * 180.0 / pi

def DegToRad(value):
	return value * pi / 180.0

def ConvertRPYToMat(psi, theta, phi):
	Matrix = np.array([[cos(phi)*cos(theta), cos(phi)*sin(theta)*sin(psi) - sin(phi)*cos(psi), cos(phi)*sin(theta)*cos(psi)+sin(phi)*sin(psi)],
						[sin(phi)*cos(theta), sin(phi)*sin(theta)*sin(psi) + cos(phi)*cos(psi), sin(phi)*sin(theta)*cos(psi)-cos(phi)*sin(psi)],
						[-sin(theta), cos(theta)*sin(psi), cos(theta)*cos(psi)]])
	return Matrix

def ConvertMatToRPY(Matrix):
	a11 = Matrix[0][0]
	a21 = Matrix[1][0]
	a31 = Matrix[2][0]
	a32 = Matrix[2][1]
	a33 = Matrix[2][2]
	theta = -asin(a31)
	psi = atan2(a32/cos(theta), a33/cos(theta))
	phi = atan2(a21/cos(theta), a11/cos(theta))
	return psi, theta, phi

def DHMatrix(q, d, a, alpha):
	return np.array([[cos(q), -cos(alpha)*sin(q), sin(alpha)*sin(q), a*cos(q)],
					[sin(q), cos(alpha)*cos(q), -sin(alpha)*cos(q), a*sin(q)],
					[0, sin(alpha), cos(alpha), d],
					[0,0,0,1]])
def SmartDegSubstraction(v2, v1):
	# v2 and v1 is numpy array
	size = v2.shape[0]
	Result = v2 - v1
	for i in np.arange(size):
		if Result[i] > pi:
			Result[i] -= 2 * pi
		elif (Result[i] < -pi):
			Result[i] += 2 * pi
	return Result

def LoadGCode(filename, offsetx, offsety, offsetz):
	file_obj = open(filename, "r")
	list_of_line = file_obj.readlines()
	list_of_gcode = []
	act = int(0)
	list_of_point = []
	for ls in list_of_line:
		if ("G1" in ls) or ("M300" in ls):
			list_of_gcode.append(ls)
		if "(end of print job)" in ls:
			break

	for ls in list_of_gcode:
		if "M300" in ls:
			ls_split_space = ls.split(" ")
			if ls_split_space[1] == "S30.00":
				act = int(1)
			elif(ls_split_space[1] == "S50.00"):
				act = int(0)
		if "G1" in ls:
			ls_split_space = ls.split(" ")
			x = offsetx + float(ls_split_space[1][1:])
			y = offsety + float(ls_split_space[2][1:])
			z = offsetz
			list_of_point.append([x, y, z, act])
	return np.asarray(list_of_point)