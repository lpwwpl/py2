import numpy as np 
import math
import os
import sys
import cv2
from sklearn.preprocessing import normalize
from scipy.spatial.transform import Rotation as R
import torch
from PySide2 import QtCore, QtGui, QtWidgets
import re
def executable_path():
    return os.path.dirname(sys.argv[0])

# 旋转矩阵转旋转矢量
def rm2rv(R):
    theta = np.arccos((R[0][0] + R[1][1] + R[2][2] - 1) / 2)
    K = (1 / (2 * np.sin(theta))) * np.asarray([R[2][1] - R[1][2], R[0][2] - R[2][0], R[1][0] - R[0][1]])
    r = theta * K
    r1 = [R[0][3], R[1][3], R[2][3], r[0], r[1], r[2]]
    return r1


# 旋转矢量转旋转矩阵
def rv2rm(rv):
    rx = rv[3]
    ry = rv[4]
    rz = rv[5]
    np.seterr(invalid='ignore')
    theta = np.linalg.norm([rx, ry, rz])
    kx = rx / theta
    ky = ry / theta
    kz = rz / theta

    c = np.cos(theta)
    s = np.sin(theta)
    v = 1 - c

    R = np.zeros((4, 4))
    R[0][0] = kx * kx * v + c
    R[0][1] = kx * ky * v - kz * s
    R[0][2] = kx * kz * v + ky * s
    R[0][3] = rv[0]

    R[1][0] = ky * kx * v + kz * s
    R[1][1] = ky * ky * v + c
    R[1][2] = ky * kz * v - kx * s
    R[1][3] = rv[1]

    R[2][0] = kz * kx * v - ky * s
    R[2][1] = kz * ky * v + kx * s
    R[2][2] = kz * kz * v + c
    R[2][3] = rv[2]
    R[3][3] = 1

    return R


def quaternion_to_euler_fc(params):
    qx = params[0]
    qy = params[1]
    qz = params[2]
    qw = params[3]
    t0=+2.0 * (qw * qx+qy * qz)
    t1=+1.0-2.0 * (qx * qx+qy * qy)
    roll=math.degrees(math.atan2(t0,t1))

    t2=+2.0 * (qw * qy-qz * qx)
    t2=+1.0if t2 > +1.0 else t2
    t2=-1.0if t2 < -1.0 else t2
    pitch=math.degrees(math.asin(t2))

    t3=+2.0 * (qw * qz+qx * qy)
    t4=+1.0-2.0 * (qy * qy+qz * qz)
    yaw=math.degrees(math.atan2(t3,t4))

    return roll,pitch,yaw


def quaternion_to_euler(params):
    qx = params[0]
    qy = params[1]
    qz = params[2]
    qw = params[3]
    t0=+2.0 * (qw * qx+qy * qz)
    t1=+1.0-2.0 * (qx * qx+qy * qy)
    roll=math.atan2(t0,t1)

    t2=+2.0 * (qw * qy-qz * qx)
    t2=+1.0if t2 > +1.0 else t2
    t2=-1.0if t2 < -1.0 else t2
    pitch=math.asin(t2)

    t3=+2.0 * (qw * qz+qx * qy)
    t4=+1.0-2.0 * (qy * qy+qz * qz)
    yaw=math.atan2(t3,t4)

    return roll,pitch,yaw

# 旋转矩阵转rpy欧拉角
def rm2rpy(R):
    # sy = np.sqrt(R[0][0] * R[0][0] + R[1][0] * R[1][0])
    sy = np.sqrt(R[2][1] * R[2][1] + R[2][2] * R[2][2])
    singular = sy < 1e-6

    if not singular:
        rotatex = np.arctan2(R[2][1], R[2][2])
        rotatey = np.arctan2(-R[2][0], sy)
        rotatez = np.arctan2(R[1][0], R[0][0])
    else:
        rotatex = np.arctan2(-R[1][2], R[1][1])
        rotatey = np.arctan2(-R[2][0], sy)
        rotatez = 0

    return np.asarray([R[0][3], R[1][3], R[2][3], rotatex, rotatey, rotatez])


# rpy转旋转矩阵
def rpy2rm(rpy):
    # Rx = np.zeros((3, 3), dtype=rpy.dtype)
    # Ry = np.zeros((3, 3), dtype=rpy.dtype)
    # Rz = np.zeros((3, 3), dtype=rpy.dtype)

    R0 = np.zeros((4, 4))

    x = rpy[0]
    y = rpy[1]
    z = rpy[2]
    thetaX = rpy[3]
    thetaY = rpy[4]
    thetaZ = rpy[5]

    cx = np.cos(thetaX)
    sx = np.sin(thetaX)

    cy = np.cos(thetaY)
    sy = np.sin(thetaY)

    cz = np.cos(thetaZ)
    sz = np.sin(thetaZ)

    R0[0][0] = cz * cy
    R0[0][1] = cz * sy * sx - sz * cx
    R0[0][2] = cz * sy * cx + sz * sx
    R0[0][3] = x
    R0[1][0] = sz * cy
    R0[1][1] = sz * sy * sx + cz * cx
    R0[1][2] = sz * sy * cx - cz * sx
    R0[1][3] = y
    R0[2][0] = -sy
    R0[2][1] = cy * sx
    R0[2][2] = cy * cx
    R0[2][3] = z
    R0[3][3] = 1
    return R0


def rv2rpy(rv):
    R = rv2rm(rv)
    rpy = rm2rpy(R)
    return rpy


def rpy2rv(rpy):
    R = rpy2rm(rpy)
    rv = rm2rv(R)
    return rv


def rpy2qt(rpy):
    rx = rpy[3]
    ry = rpy[4]
    rz = rpy[5]
    x = np.cos(ry / 2) * np.cos(rz / 2) * np.sin(rx / 2) - np.sin(ry / 2) * np.sin(rz / 2) * np.cos(rx / 2)
    y = np.sin(ry / 2) * np.cos(rz / 2) * np.cos(rx / 2) + np.cos(ry / 2) * np.sin(rz / 2) * np.sin(rx / 2)
    z = np.cos(ry / 2) * np.sin(rz / 2) * np.cos(rx / 2) - np.sin(ry / 2) * np.cos(rz / 2) * np.sin(rx / 2)
    w = np.cos(ry / 2) * np.cos(rz / 2) * np.cos(rx / 2) + np.sin(ry / 2) * np.sin(rz / 2) * np.sin(rx / 2)
    return [x, y, z, w]




####################################



def calAxisPoints(centroid, rotation, length, point_num):
    x = rotation[0].T
    y = rotation[1].T
    z = rotation[2].T
    dis = length / point_num

    xAxisPoints = []
    yAxisPoints = []
    zAxisPoints = []
    for i in range(0, point_num):
        xAxisPoints.append(centroid + x * (i * dis))
        yAxisPoints.append(centroid + y * (i * dis))
        zAxisPoints.append(centroid + z * (i * dis))
    # xAxisPoints = [(centroid+x*(i*dis)) for i in range(1, point_num+1)]
    # yAxisPoints = [(centroid+y*(i*dis)) for i in range(1, point_num+1)]
    # zAxisPoints = [(centroid+z*(i*dis)) for i in range(1, point_num+1)]
    axisPoints = np.vstack([xAxisPoints, yAxisPoints, zAxisPoints])
    axisPointsColor = np.vstack((np.tile([255, 0, 0], (point_num, 1)), np.tile([0, 255, 0], (point_num, 1)),
                                 np.tile([0, 0, 255], (point_num, 1))))
    return [axisPoints, axisPointsColor]


def calImageAxis(centroid, rotation, length, cameraIntrinsics):

    o = centroid.reshape(3, 1)

    xyz = rotation * length + np.tile(centroid, (3, 1)).T

    oxyz = np.hstack((o, xyz))

    imageAxisPix = np.dot(cameraIntrinsics, oxyz)
    # print("imageAxisPix",imageAxisPix)
    imageAxisPix = imageAxisPix / np.tile(imageAxisPix[2], (3, 1))
    return imageAxisPix[0:2].T

class HMUtil:
    def convertXYZABCtoHM(func):
        def wrapper(*args, **kwargs):
            funcout = func(*args, **kwargs)
            [x,y,z,a,b,c] = funcout

            ca = math.cos(a)
            sa = math.sin(a)
            cb = math.cos(b)
            sb = math.sin(b)
            cc = math.cos(c)
            sc = math.sin(c)    
            H = np.array([[cb*cc, cc*sa*sb - ca*sc, sa*sc + ca*cc*sb, x],[cb*sc, ca*cc + sa*sb*sc, ca*sb*sc - cc*sa, y],[-sb, cb*sa, ca*cb, z],[0,0,0,1]])
            return H
        return wrapper

    def convertHMtoXYZABC(func):
        def wrapper(*args, **kwargs):
            H = args[0]
            x = H[0,3]
            y = H[1,3]
            z = H[2,3]
            if (H[2,0] > (1.0 - 1e-10)):
                b = -math.pi/2
                a = 0
                c = math.atan2(-H[1,2],H[1,1])
            elif H[2,0] < -1.0 + 1e-10:
                b = math.pi/2
                a = 0
                c = math.atan2(H[1,2],H[1,1])
            else:
                b = math.atan2(-H[2,0],math.sqrt(H[0,0]*H[0,0]+H[1,0]*H[1,0]))
                c = math.atan2(H[1,0],H[0,0])
                a = math.atan2(H[2,1],H[2,2])    
            funcout = func([x, y, z, a, b, c])
            return funcout
        return wrapper

    @staticmethod
    @convertXYZABCtoHM
    def convertXYZABCtoHMDeg(xyzabc):
        [x,y,z,a,b,c] = xyzabc
        a = a*math.pi/180
        b = b*math.pi/180
        c = c*math.pi/180
        return [x, y, z, a, b, c]

    @staticmethod
    @convertXYZABCtoHM
    def convertXYZABCtoHMRad(xyzabc):
        return xyzabc

    @staticmethod
    @convertHMtoXYZABC
    def convertHMtoXYZABCDeg(xyzabc):
        [x,y,z,a,b,c] = xyzabc
        return [x, y, z, a*180/math.pi, b*180/math.pi, c*180/math.pi]

    '''
    HM =        R(3x3)     d(3x1)
                0(1x3)     1(1x1)

    HMInv =     R.T(3x3)   -R.T(3x3)*d(3x1)
                0(1x3)     1(1x1)

                (R^-1 = R.T)
    '''
    @staticmethod
    def inverseHM(H):
        rot = H[0:3, 0:3]
        trs = H[0:3, 3]

        HMInv = np.zeros([4, 4], dtype=np.float64)
        HMInv[0:3, 0:3] = rot.T
        HMInv[0:3, 3] = (-1.0)*np.dot(rot.T, trs)
        HMInv[3, 0:4] = [0.0, 0.0, 0.0, 1.0]
        return HMInv
    
    @staticmethod
    def invH(H):
        Hout = H.T
        Hout[3,0:3] = np.zeros([[0,0,0]])
        Hout[0:3,3] = (Hout[0:3,0:3]*H[0:3,3])*(-1)
        return Hout    
    
    @staticmethod
    def makeHM(rot, trans):
        HM = np.zeros([4, 4], dtype=np.float64)
        HM[0:3, 0:3] = rot
        HM[0:3, 3] = trans.reshape(-1)
        HM[3, 0:4] = [0.0, 0.0, 0.0, 1.0]
        return HM



def recalRotation(rotation):
    z = rotation[:, 2]
    z = normalize([z], norm='l2')
    z_cam = np.asarray([0, 0, 1])

    # if (z_cam * z.T < 0):
    if float(np.dot(z_cam, z.T)) < 0:
        z = z * (-1)

    crossProd = np.cross(z_cam, z)
    if (np.linalg.norm(crossProd, ord=2) == 0):
        return np.eye(3)

    rotAxis = normalize(crossProd, norm='l2')
    rotAngle = math.acos(np.dot(z_cam, z.T))

    # M = makehgtform('axisrotate',rotAxis,rotAngle)
    # x = np.arange(1,3)
    # y = np.arange(1,3)
    # rotation=M(x,y)
    r1 = R.from_rotvec(rotAxis)
    r2 = R.from_euler('xyz', [rotAngle, rotAngle, rotAngle])
    rotation = (r1 * r2).as_matrix()
    return rotation[0]


def isRotationMatrix(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6


def rotationMatrixToEulerAngles(R):
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def cluster_filter(cluster, idx):
    # [value, ~] = sort(cluster, 1, 'descend');
    value = cluster[:]
    value = np.sort(value, axis=0)[::-1]  # descend
    sz = cluster.shape
    mid_point = value[round(sz[0] / 2), :]
    temp = np.subtract(cluster, mid_point)
    dist = 0
    for i in range(0, sz[1]):
        # xx = temp[:,i]
        dist = dist + (np.square(temp[:, i]))

    dist = np.sqrt(dist)
    index = np.argsort(dist)
    value = np.sort(dist)
    delta_dist = []

    if len(value) == 1:
        print(value)
        del_index = index[0+1:]
        del_idx = idx[del_index]
        return del_idx
    else:

        for i in range(0, len(value) - 1):
            delta_dist.append(value[i + 1] - value[i])

        delete_point_found = False
        for i in range(0, len(delta_dist)):
            if delta_dist[i] > 1e-2:
                delete_point_found = True
                break
        if delete_point_found:
            del_index = index[i + 1:]
            del_idx = idx[del_index]
            return del_idx
        else:
            del_index = []
            return []



def image_wrapper(colorImg, depthImg):
    mean = 0.456  # np.array([0.485, 0.456, 0.406])
    std = 0.225  # np.array([0.229, 0.224, 0.225])

    # change image data type
    grayImg = cv2.cvtColor(colorImg, cv2.COLOR_RGB2GRAY).astype('float32')
    depthImg = depthImg.astype('float32')

    # pre-process gray image
    grayImg = (grayImg / 255.0 - mean) / std
    grayImg = grayImg[:, :, np.newaxis]

    # pre-process depth image

    depthImg = depthImg / 10000
    depthImg = np.clip(depthImg, 0.0, 2.0)  # Depth range limit
    depthImg = (depthImg - mean) / std
    depthImg = depthImg[:, :, np.newaxis]

    # form data
    data = np.concatenate((grayImg, depthImg, depthImg), axis=2)

    # reshape dimensions
    data = torch.tensor(data, dtype=torch.float32)
    data = data.permute(2, 0, 1)  # dimensions [3,480,640]
    data = data.unsqueeze(dim=0)

    return data





def sleepWithEventLoop(timeout):
    if timeout >0 :
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(timeout,loop.quit)
        loop.exec_()


def check_ip(ipAddr):
    compile_ip=re.compile('^(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|[1-9])\.(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|\d)\.(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|\d)\.(1\d{2}|2[0-4]\d|25[0-5]|[1-9]\d|\d)$')
    if compile_ip.match(ipAddr):
        return True
    else:
        return False





###############################################################################
# Test Codes                                                                         
###############################################################################
if __name__ == '__main__':

    # prepare test data - xyzabc and homogeneous matrix
    xyzabc = [-0.09064515960284498, -0.6685702677827611, 0.07567205103501873, -176.08612962588248, -0.6780892276157752, 130.42940636697082]

    hm = np.array(([[-0.6484945560472715,  0.7588883777053825,     -0.05952512881754177,   -0.09061736124264554], 
                    [0.7611366063389998,    0.6475911239041057,     -0.03601114732090388,   -0.6687733681469619], 
                    [0.01121950390181825,   -0.06865978753469798,   -0.9975770427931304,    0.07609749637689898], 
                    [0,                     0,                      0,                      1]]), dtype=np.float64
                )

    # 0. print original xyzabc and hm
    print('Test Data')
    print("XYZABC = ")
    print(xyzabc)
    print("HM = ")
    print(hm)
    print()
    
    # 1.  convert XYZABC(degree) to a homogeneous matrix
    print('XYZABC --> HM')
    hmcal = HMUtil.convertXYZABCtoHMDeg(xyzabc)
    print('Converted HM')
    print(hmcal)
    print()
    
    # 2. convert a homogeneous matrix to XYZABC(degree)
    print('Converted XYZABC')
    xyzabc_calc = HMUtil.convertHMtoXYZABCDeg(hmcal)
    print(xyzabc_calc)
    print()

    # 3. test inverseHM
    hmpinv = np.linalg.pinv(hm)
    print('Pseudo Inverse')
    print(hmpinv)
    print('Check if inverse is available')
    print(np.dot(hmpinv, np.array([-0.09064515960284498, -0.6685702677827611, 0.07567205103501873, 1], dtype=np.float64).T))
    print()

    hminv = HMUtil.inverseHM(hm)
    print('Homogeneous Inverse')
    print(hminv)
    print('Check if inverse is available')
    print(np.dot(hminv, np.array([-0.09064515960284498, -0.6685702677827611, 0.07567205103501873, 1], dtype=np.float64).T))

