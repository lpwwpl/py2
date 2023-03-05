from PySide2 import QtCore, QtGui, QtWidgets
import rsData
from UtilSet import *
import cv2
import torch
import torch.nn as nn
import torchvision.models as models

from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as sch


import time
from ThreeDSurfaceGraphWindow import ThreeDSurfaceGraphWindowDlg

from rsData import EModet
from rsData import ERunning
pca = PCA()
timestamp = None


def inference(inferImgSavePath, colorImg, deptImg):
    data = image_wrapper(colorImg, deptImg)

    with torch.no_grad():
        # move data to device
        data = data.to(rsData.torch_device)

        # forward inference
        t1 = time.time()
        out = rsData.net(data, phase=2)  # out size [1,3,60,80]
        t2 = time.time()

        # calculate inference image
        out = rsData.softmax(out)
        out = torch.nn.functional.interpolate(out, scale_factor=2, mode='bilinear', align_corners=True)
        inferImg = out[0, 1, :, :]  # size [120,160]
        # inferImg = (inferImg * 255).clamp_(0, 255).round().numpy().astype('uint8')
        # .cuda()
        # inferImg = (inferImg * 255).clamp_(0, 255).round().cuda().data.cpu().numpy().astype('uint8')
        inferImg = (inferImg * 255).clamp_(0, 255).round().numpy().astype('uint8')
        inferImg = cv2.resize(inferImg, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)  # size [480,640]

    # print inference time usage
    rsData.log.debug('Inference service time usage (in secs): %.3f' % (t2 - t1))

    cv2.imwrite(inferImgSavePath, inferImg)

class RevNet(nn.Module):
    def __init__(self, baseNet='resnet50', pretrained=False):
        super(RevNet, self).__init__()

        if baseNet == 'resnet101':
            net_imported = models.resnet101(pretrained=pretrained)
        elif baseNet == 'resnet34':
            net_imported = models.resnet34(pretrained=pretrained)
        else:
            net_imported = models.resnet50(pretrained=pretrained)

        if baseNet == 'resnet34':
            out_size = 512 / 2
        else:
            out_size = 2048 / 2

        self.resTower1 = nn.Sequential(*list(net_imported.children())[:-3])
        # self.resTower2 = nn.Sequential(*list(net_imported.children())[:-3])
        self.conv_e1 = nn.Conv2d(int(out_size), int(out_size / 4), kernel_size=1, stride=1, bias=False)  # 2048,512
        self.conv_e2 = nn.Conv2d(int(out_size / 4), int(out_size / 16), kernel_size=1, stride=1, bias=False)  # 512,128
        self.conv_e3 = nn.Conv2d(int(out_size / 16), 3, kernel_size=1, stride=1, bias=False)  # 128,3
        self.upsample1 = nn.UpsamplingBilinear2d(scale_factor=2)

        n = self.conv_e1.kernel_size[0] * self.conv_e1.kernel_size[1] * self.conv_e1.out_channels
        self.conv_e1.weight.data.normal_(0, math.sqrt(2. / n))
        n = self.conv_e2.kernel_size[0] * self.conv_e2.kernel_size[1] * self.conv_e2.out_channels
        self.conv_e2.weight.data.normal_(0, math.sqrt(2. / n))
        n = self.conv_e3.kernel_size[0] * self.conv_e3.kernel_size[1] * self.conv_e3.out_channels
        self.conv_e3.weight.data.normal_(0, math.sqrt(2. / n))

    def forward(self, x, phase):
        if phase == 1:
            with torch.no_grad():
                x = self.resTower1(x)
        else:
            x = self.resTower1(x)
        x = self.conv_e1(x)
        x = self.conv_e2(x)
        x = self.conv_e3(x)
        x = self.upsample1(x)

        return x


class KCore(QtCore.QThread):
    pic_signal = QtCore.Signal()

    def __init__(self, parent=None):
        super(KCore, self).__init__(parent)
        self.working = True
        self.Tool2Base_Shooting = [0.000, -0.000, 0.36010000000000003 ,0, 0, 0]
        self.path = ""

    # 机器人拍摄位姿计算(指定拍摄位姿)
    def Fun_Eyeinhand_Shooting(self,flag):
        if flag == False:
            # qua =[0,0,0,1]
            # a=quaternion_to_euler_fc(qua)
            # print(a)
            self.Tool2Base_Shooting = [0.000, -0.000, 0.36010000000000003 ,0, 0, 0]
        else:
            self.Tool2Base_Shooting = [-0.26879975 , -0.32541972 , 0.1903955 , 0.009104541870731944,
                                  3.130781700067779,
                                  0.05941563583525477]
        return self.Tool2Base_Shooting

    def move_to_joints(self,joints,tool_acc, tool_vel,active=False,wait_for_complete=True):
        if rsData.r:
            rsData.r_ready = False
            rsData.r.set_joints(joints)
            if wait_for_complete:
                while True:
                    self.move_complete_timeout()
                    if rsData.r_ready or not rsData.r:
                        break
                    sleepWithEventLoop(100)
        time.sleep(1)

    def move_complete_timeout(self):
        moveComplete=0
        if rsData.r:
            moveComplete=rsData.r.get_move_complete()
        if moveComplete == 1:
            print("moveComplete,",moveComplete)
            rsData.r_ready = True
        else:
            rsData.r_ready = False



    def move_to_tcp(self, tcp, tool_acc, tool_vel,active=False,wait_for_complete=True):
        cartesian=[[],[]]
        cartesian[0]=tcp[:3]
        cartesian[1]=rpy2qt(tcp)
        # cartesian = [26.66, 17.75, -3.53, -22.42, 87.13, 184.75]
        if rsData.r:
            rsData.r_ready = False
            rsData.r.set_cartesian(cartesian)
            if wait_for_complete:
                while True:
                    self.move_complete_timeout()
                    if rsData.r_ready or not rsData.r:
                        break
                    sleepWithEventLoop(100)
        time.sleep(1)

    def Fun_Suction_Grip(self):
        if rsData.r:
            rsData.r.set_gripper_on()

    def Fun_Suction_Release(self):
        if rsData.r:
            rsData.r.set_gripper_off()

    def text_read(self):
        Cam2Base_rm = [[9.99994199e-01,  9.47682884e-04,  3.27178992e-03,  5.14732905e-02],
                       [9.29586803e-04 ,- 9.99984288e-01 , 5.52803946e-03,  2.11069855e-03],
                       [3.27697734e-03 ,- 5.52496598e-03 ,- 9.99979368e-01, 3.58851006e-02],
                       [0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]]
        return Cam2Base_rm

    def Fun_Tool2Base_Suction_Interim(self,Object2Cam):
        # Object2Cam = client_srv()
        Object2Cam_rm = rpy2rm(Object2Cam)
        Cam2Tool_rm = self.text_read()
        # Tool2Base = self.get_current_tcp()
        Tool2Base_rm = rv2rm(self.Tool2Base_Shooting)
        Object2Base = Tool2Base_rm.dot(Cam2Tool_rm).dot(Object2Cam_rm)
        Tool2Base_Suction_rm = Object2Base
        Tool2Base_Suction = rm2rv(Tool2Base_Suction_rm)  # 抓取位姿
        Move_rm = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -0.05 ], [0, 0, 0, 1]])
        Tool2Base_Interim1_rm = Tool2Base_Suction_rm.dot(Move_rm)
        Tool2Base_Interim1 = rm2rv(Tool2Base_Interim1_rm)  # 过渡位姿
        Tool2Base_Interim2 = Tool2Base_Interim1
        return Tool2Base_Suction, Tool2Base_Interim1, Tool2Base_Interim2

    def suction_process(self):
        target_tcp_shooting = self.Fun_Eyeinhand_Shooting(False)  # 机器人拍摄位姿计算
        # self.move_to_tcp(target_tcp_shooting, 1.5, 1)  # 移动至机器人拍摄位姿
        self.move_to_joints([2.03, 35.28, -8.94, -1.11, 63.70, 90.94],1.5, 1)

        Object2Cam = None
        try:
            Object2Cam = self.client_srv()
        except Exception as e:
            rsData.log.error(e)
        if Object2Cam:
            rsData.log.debug("obj2cam:{}".format(Object2Cam))
            temp=Object2Cam[0]
            Object2Cam[0]=Object2Cam[1]
            Object2Cam[1]=temp
            [target_tcp_suction, target_tcp_interim1,
             target_tcp_interim2] = self.Fun_Tool2Base_Suction_Interim(Object2Cam)  # 机器人抓取位姿和过渡位姿计算
            rsData.log.debug("target_tcp_suction:{}".format(target_tcp_suction))
            ##self.move_to_tcp(target_tcp_interim1, 1.5, 1)  # 移动至机器人抓取过渡位姿
            self.move_to_joints([1.74, 54.08, -32.18, -0.99, 78.10, 90.54],1.5, 1)

            ##self.move_to_tcp(target_tcp_suction, 0.5, 0.5,True)  # 移动至机器人抓取位姿
            self.move_to_joints([7.83, 30.29, -19.04, -1.01, 78.89, 76.29],1.5, 1)
            # self.Fun_Suction_Grip()
            ##self.move_to_tcp(target_tcp_interim2, 0.5, 1)  # 移动至机器人过渡位姿
            # [target_tcp_release, target_tcp_interim_1, target_tcp_interim_2] = self.Fun_Eyeinhand_Release(False)  # 机器人放置位姿计算
            # self.move_to_tcp(target_tcp_interim_2, 1.5, 1)  # 移动至机器人释放过渡位姿
            # self.move_to_tcp(target_tcp_release, 0.5, 0.5)  # 移动至机器人释放位姿
            # print("target_tcp_release:{}".format(target_tcp_release))
            # self.Fun_Suction_Release()
            # self.move_to_tcp(target_tcp_interim_1, 0.5, 0.5)  # 移动至机器人释放位姿

    def init_path(self):
        if self.path == "" or self.path != rsData.cur_root_path:
            file = QtCore.QFileInfo(rsData.cur_root_path)
            self.path = file.absoluteFilePath()
            intriFileName = os.path.abspath("{}/test-camera-intrinsics.txt".format(self.path))
            self.colorCamIntrinsics = np.loadtxt(intriFileName, dtype=float)

    def run(self):
        try:
            self.init_path()
        except Exception as e:
            rsData.log.error(e)
            return
        while self.working == True:
            if not rsData.r:
                break
            self.suction_process()
            time.sleep(0.5)
            # self.working =False

    def client_srv(self, prefix=None):
        self.init_path()

        if prefix:
            colorFileName = os.path.abspath( "{}/{}.color.png".format(self.path,prefix))
            deptFileName = os.path.abspath( "{}/{}.depth.png".format(self.path,prefix))
            inferFileName = os.path.abspath("{}/{}.infer.png".format(self.path,prefix))
            color_img = cv2.cvtColor(cv2.imread(colorFileName, flags=1), cv2.COLOR_BGR2RGB)
            depth_img = cv2.imread(deptFileName, flags=-1)
            # depth_img = depth_img*10000
            # depth_mat = np.zeros((480, 640), np.uint16)
            # for y in range(480):
            #     for x in range(640):
            #         depth_short = depth_img[y, x] * 10000
            #         depth_mat[y, x] = depth_short
            # depth_img = depth_mat
            # deptfilepath="picslx/temp.png"
            # cv2.imwrite(deptfilepath, depth_mat)
            # depth_img = cv2.imread(deptfilepath, flags=-1)
            # color_img=
            inferFileName = os.path.abspath("{}/{}.infer.png".format(self.path,prefix))
        else:
            color_img = rsData.color_streaming
            depth_img = rsData.depthImg

            # deptfilepath="picslx/temp.png"
            # cv.imwrite(deptfilepath, depth_mat)
            # depth_img = cv.imread(deptfilepath, flags=-1)
            # depth_img = depth_mat
            # color_img=
            inferFileName = os.path.abspath("{}/frame-{}.infer.png".format(self.path,prefix))
        # inference(inferFileName, self.cam.color_img, self.cam.depth_img)
        inference(inferFileName, color_img, depth_img)
        inferImg = cv2.imread(inferFileName, flags=-1)

        # centroids_cam, rotations, confidence_clusters = self.postProcess(False, self.cam.color_img, self.cam.depth_img,
        #                                                                  inferImg, self.colorCamIntrinsics)
        centroids_cam, rotations, confidence_clusters = self.postProcess(False, color_img, depth_img,
                                                                           inferImg, self.colorCamIntrinsics)
        Object2Cam = []

        centroids_cam_array = np.asarray(centroids_cam)
        dim = centroids_cam_array.ndim

        if (len(confidence_clusters) > 0):
            if (dim == 1):
                # EulerXYZ = rotations[0]
                # eulerAngle = rotationMatrixToEulerAngles(EulerXYZ)
                eulerAngle = [0, 0, 0]
                Object2Cam = [centroids_cam[0], centroids_cam[1], centroids_cam[2], eulerAngle[0],
                              eulerAngle[1],
                              eulerAngle[2]]
            else:
                Transitions = centroids_cam[0]
                # EulerXYZ = rotations[0]
                eulerAngle = [0, 0, 0]
                # eulerAngle = rotationMatrixToEulerAngles(EulerXYZ)
                Object2Cam = [Transitions[0], Transitions[1], Transitions[2], eulerAngle[0],
                              eulerAngle[1],
                              eulerAngle[2]]

        return Object2Cam

    def postProcess(self, ShowImages=False, colorImg=None, depthImg=None, inferImg=None, cameraIntrinsics=None):
        iheight = rsData.height
        iwidth = rsData.width
        depthImg = depthImg.astype('double') / 10000.0

        threthold_lowerbound = 0.2
        affordance_threthold = inferImg.max().astype('double') / 255.0 * 0.64

        if affordance_threthold < threthold_lowerbound:
            affordance_threthold = threthold_lowerbound
        mask_th = 255 * affordance_threthold
        mask_p = (inferImg >= mask_th)

        x = np.arange(1, iwidth + 1)
        y = np.arange(1, iheight + 1)
        [pixX, pixY] = np.meshgrid(x, y)
        pixZ = depthImg.astype(np.double)

        pixX = pixX.T
        pixX_clusters = pixX[mask_p.T]

        pixY = pixY.T
        pixY_clusters = pixY[mask_p.T]
        pixZ = pixZ.T
        pixZ_clusters = pixZ[mask_p.T]
        pixels_clusters = np.asarray([pixX_clusters, pixY_clusters])
        if mask_p.sum() == 0:
            cluster_idx = []
            cluster_count = 0
        else:
            Y = pdist(pixels_clusters.T, 'euclidean')
            Z = sch.linkage(Y, 'single', True)
            # inconsis = inconsistent(Z)
            cluster_idx = sch.fcluster(Z, t=0.6, criterion='inconsistent')  # 'Cutoff'
            cluster_count = max(cluster_idx)

        camX_clusters = (pixX_clusters - cameraIntrinsics[0, 2]) * pixZ_clusters / cameraIntrinsics[0, 0]
        camY_clusters = (pixY_clusters - cameraIntrinsics[1, 2]) * pixZ_clusters / cameraIntrinsics[1, 1]
        camZ_clusters = pixZ_clusters
        camPoints_clusters = np.asarray([camX_clusters, camY_clusters, camZ_clusters])

        camPointsColor_clusters = []
        for i in range(0, 3):
            temp = colorImg[:, :, i]
            temp = temp[mask_p]
            if len(camPointsColor_clusters) == 0:
                camPointsColor_clusters = temp
            else:
                camPointsColor_clusters = np.concatenate((camPointsColor_clusters, temp))


        camX = (pixX - cameraIntrinsics[0, 2]) * pixZ / cameraIntrinsics[0, 0]
        camY = (pixY - cameraIntrinsics[1, 2]) * pixZ / cameraIntrinsics[1, 1]
        camZ = pixZ
        camPoints = [camX, camY, camZ]

        # camPointsColor = colorImg.reshape((-1, 3), order='F')

        camPointsColor = []
        for i in range(0, 3):
            temp = colorImg[:, :, i]
            if len(camPointsColor) == 0:
                camPointsColor = temp
            else:
                camPointsColor = np.concatenate((camPointsColor, temp))

        del_idx = []
        for i in range(1, cluster_count + 1):
            idx_filter = (cluster_idx == i)
            tmp = np.asarray([camX_clusters[idx_filter], camY_clusters[idx_filter], camZ_clusters[idx_filter]])
            tmp = tmp.T
            tmp_ = np.where(idx_filter)[0]
            cluster_filter_temp = cluster_filter(tmp, tmp_)
            # lpw = np.where(idx_filter)
            # tmp1 = camPoints_clusters_T[lpw]
            # tmp2 = np.where(cluster_idx == i)
            # cluster_filter_temp = cluster_filter(tmp1, tmp2)
            # del_idx = np.concatenate((del_idx, cluster_filter_temp))

        if len(del_idx):
            del_idx = del_idx.astype('int')
        if len(del_idx) > 0:
            pixX_clusters[del_idx] = False
            pixY_clusters[del_idx] = False
            pixZ_clusters[del_idx] = False
            pixels_clusters[0][del_idx] = False
            pixels_clusters[1][del_idx] = False
            camX_clusters[del_idx] = False
            camY_clusters[del_idx] = False
            camZ_clusters[del_idx] = False
            camPoints_clusters[0][del_idx] = False
            camPoints_clusters[1][del_idx] = False
            camPoints_clusters[2][del_idx] = False
            cluster_idx[del_idx] = False

        clusters = np.zeros([iheight, iwidth], 'int')
        for i in range(1, cluster_count + 1):
            temp1 = pixels_clusters[0][cluster_idx == i]
            temp2 = pixels_clusters[1][cluster_idx == i]
            temp = (temp2 - 1) * iwidth + temp1
            clusters.flat[temp] = i * 255.0 / cluster_count

        rsData.log.debug('Number of clusters: %d', cluster_count)
        for i in range(1, cluster_count + 1):
            rsData.log.debug('Number of points in cluster' + str(i) + ":" + str(sum(cluster_idx == i)))

        # infer_clusters = inferImg.T[mask_p.T]
        infer_clusters = inferImg[mask_p]
        confidence_clusters = []
        for i in range(1, cluster_count + 1):
            idx_cluster = (cluster_idx == i)
            x = infer_clusters[idx_cluster]
            if len(infer_clusters[idx_cluster]) > 0:
                temp = infer_clusters[cluster_idx == i]
                confidence_clusters.append(max(temp))
        confidence_idx = np.argsort(confidence_clusters)  # [0 2 1]
        confidence_clusters = np.sort(confidence_clusters)[::-1]  # descend

        centroids_cam = []
        for i in range(1, cluster_count + 1):
            idx_filter = (cluster_idx[::] == i)
            cent_tmp1 = camPoints_clusters[0][idx_filter].mean()
            cent_tmp2 = camPoints_clusters[1][idx_filter].mean()
            cent_tmp3 = camPoints_clusters[2][idx_filter].mean()
            if len(centroids_cam) == 0:
                centroids_cam = [cent_tmp1, cent_tmp2, cent_tmp3]
                centroids_cam = np.asarray(centroids_cam)
            else:
                centroids_cam = np.row_stack((centroids_cam, [cent_tmp1, cent_tmp2, cent_tmp3]))
        k = []

        # if centroids_cam != []:
        # row,col= centroids_cam.shape
        #
        if len(centroids_cam) > 0:


            dim = centroids_cam.ndim

            if dim > 1:
                for i in range(centroids_cam.shape[0]):
                    mask_c1 = centroids_cam[i, 2] > 0.6
                    mask_c2 = centroids_cam[i, 2] <= 0
                    mask_c = mask_c1 | mask_c2
                    if mask_c:
                        k.append(i)
                        centroids_cam[i, :] = [0, 0, 0]
            else:
                mask_c1 = centroids_cam[2] > 0.6
                mask_c2 = centroids_cam[2] <= 0
                mask_c = mask_c1 | mask_c2
                if mask_c:
                    k.append(i)
                    centroids_cam[0] = 0
                    centroids_cam[1] = 0
                    centroids_cam[2] = 0

        rotations = []  # 3*3*3

        for i in range(1, cluster_count + 1):
            idx_filter = (cluster_idx[::] == i)
            cent_tmp1 = camPoints_clusters[0][idx_filter]
            cent_tmp2 = camPoints_clusters[1][idx_filter]
            cent_tmp3 = camPoints_clusters[2][idx_filter]
            temp = np.asarray([cent_tmp1, cent_tmp2, cent_tmp3])
            # rotation = pca.fit(temp).transform(temp)
            # rotation = recalRotation(rotation)
            # if rotation[] == 3:
            rotation = [[-0.5082,0.28273 ,0.81351],[0.71813 ,  0.66054 ,  0.21905], [-0.47543 ,  0.69552 , -0.53872]]
            rotation = np.asarray(rotation)
            rotations.append(rotation)
            # else:
            #     k.append(i)
            #     centroids_cam[i,:]=[0, 0, 0]

            # axis: red for x, green for y, blue for z
        axisPoints = []
        axisPointsColor = []
        for i in range(0, cluster_count):
            [axisPoints_tmp, axisPointsColor_tmp] = calAxisPoints(centroids_cam[i], rotations[i], 0.06, 50)
            axisPoints.append(axisPoints_tmp)
            axisPointsColor.append(axisPointsColor_tmp)

        axisPoints = np.asarray(axisPoints)
        axisPointsColor = np.asarray(axisPointsColor)

        k = list(set(k))
        if len(k) > 0:
            for j in k:
                j=j+1
                idx_filter = (cluster_idx == j)
                del_idx = np.where(idx_filter)
                pixX_clusters[del_idx] = False
                pixY_clusters[del_idx] = False
                pixZ_clusters[del_idx] = False
                pixels_clusters[0][del_idx] = False
                pixels_clusters[1][del_idx] = False
                camX_clusters[del_idx] = False
                camY_clusters[del_idx] = False
                camZ_clusters[del_idx] = False
                camPoints_clusters[0][del_idx] = False
                camPoints_clusters[1][del_idx] = False
                camPoints_clusters[2][del_idx] = False
                cluster_idx[del_idx] = False


        centroids_cam = np.asarray(centroids_cam)
        dim = centroids_cam.ndim
        try:
            if (dim == 1 and len(centroids_cam) > 0):
                if(centroids_cam[0]*centroids_cam[1]*centroids_cam[2] != 0):
                    imageAxisPix = calImageAxis(centroids_cam, rotations[0], 0.06, cameraIntrinsics)
                    # cv2.circle(colorImg,(int(imageAxisPix[0,0]), int(imageAxisPix[0,1])), 5, (255,0,0))
                    cv2.line(colorImg, (int(imageAxisPix[0, 0]), int(imageAxisPix[0, 1])),
                             (int(imageAxisPix[1, 0]), int(imageAxisPix[1, 1])), (255, 0, 0))
                    cv2.line(colorImg, (int(imageAxisPix[0, 0]), int(imageAxisPix[0, 1])),
                             (int(imageAxisPix[2, 0]), int(imageAxisPix[2, 1])), (0, 255, 0))
                    cv2.line(colorImg, (int(imageAxisPix[0, 0]), int(imageAxisPix[0, 1])),
                             (int(imageAxisPix[3, 0]), int(imageAxisPix[3, 1])), (0, 0, 255))
            else:
                for i in range(centroids_cam.shape[0]):
                    if (centroids_cam[i][0] * centroids_cam[i][1] * centroids_cam[i][2] != 0):
                        imageAxisPix = calImageAxis(centroids_cam[i], rotations[i], 0.06, cameraIntrinsics)
                        # cv2.circle(colorImg,(int(imageAxisPix[0,0]), int(imageAxisPix[0,1])), 5, (255,0,0))
                        cv2.line(colorImg, (int(imageAxisPix[0, 0]), int(imageAxisPix[0, 1])),
                                 (int(imageAxisPix[1, 0]), int(imageAxisPix[1, 1])), (255, 0, 0))
                        cv2.line(colorImg, (int(imageAxisPix[0, 0]), int(imageAxisPix[0, 1])),
                                 (int(imageAxisPix[2, 0]), int(imageAxisPix[2, 1])), (0, 255, 0))
                        cv2.line(colorImg, (int(imageAxisPix[0, 0]), int(imageAxisPix[0, 1])),
                                 (int(imageAxisPix[3, 0]), int(imageAxisPix[3, 1])), (0, 0, 255))
        except Exception as e:
            rsData.log.error(e)
        # rsData.sem.tryAcquire()
        rsData.colorImg = colorImg
        rsData.depthImg = depthImg
        rsData.inferImg = inferImg
        rsData.clusters = clusters
        rsData.pc1 = camPoints
        rsData.pc2 = camPoints_clusters
        rsData.axisPoints = axisPoints
        rsData.axisPointsColor = axisPointsColor
        rsData.camPointsColor = camPointsColor
        rsData.camPointsColor_clusters = camPointsColor_clusters
        # rsData.sem.release()
        self.pic_signal.emit()
        # rotations = np.array(rotations)
        # if len(k) > 0:
        #     for i in range(0, len(k)):
        #         dim = rotations.ndim
        #         print(k)
        #         if dim ==3:
        #             print(rotations)
        #             rotations[k[i],:, :] = [0]
        #             centroids_cam[k[i], :] = [0]
        #         else:
        #             rotations[k[i],:] = [0]
        #             centroids_cam[k[i]] = [0]
        # cluster_count = centroids_cam.shape(0)
        # [confidence_clusters, confidence_idx] = sort(confidence_clusters, 'descend');

        return centroids_cam, rotations, confidence_clusters


class ImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.m_pixmap = None
        self.m_Rect = QtCore.QRect(0,0,self.width(),self.height())
        self.width_pic = rsData.width
        self.height_pic = rsData.height
        self.radio = 1.33333

    def setImage(self,image):
        self.m_pixmap = QtGui.QPixmap.fromImage(image)
        imageWidth = self.m_pixmap.size().width()
        imageHeight = self.m_pixmap.size().height()
        # xRate = imageWidth*1.0/self.width()
        # yRate = imageHeight*1.0/self.height()
        # displayWidth = imageWidth*1.0/max(xRate,yRate)
        # displayHeigth = imageHeight*1.0/max(xRate,yRate)

        displayWidth = self.width_pic
        displayHeigth = self.height_pic
        # newW = w
        # newH = newW/self.radio
        #
        # if newH>=h:
        #     pass
        # else:
        #     newH = h
        #     newW = newH*self.radio
        # self.scale = newW/rsData.width
        # self.width_pic = newW
        # self.height_pic = newH

        self.m_Rect = QtCore.QRect((self.width()-displayWidth)/2,(self.height()-displayHeigth)/2,displayWidth,displayHeigth)
        self.m_pixmap = self.m_pixmap.scaled(self.m_Rect.width(),self.m_Rect.height(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation)
        self.update()

    def resizeEvent(self, event):
        w = self.width()
        h = self.height()

        newW = w
        newH = newW / self.radio

        if newH >= h:
            pass
        else:
            newH = h
            newW = newH * self.radio
        self.scale = newW / rsData.width
        self.width_pic = newW
        self.height_pic = newH

        self.update()

    def paintEvent(self, event):
        opt = QtWidgets.QStyleOption
        # opt.init()
        painter = QtGui.QPainter(self)

        # self.style().drawPrimitive(QtWidgets.QStyle.PE_Widget,opt,painter)
        if not self.m_pixmap.isNull():
            brush = QtGui.QBrush(QtCore.Qt.transparent)
            painter.setBackground(brush)
            painter.eraseRect(self.m_Rect)
            painter.drawPixmap(self.m_Rect,self.m_pixmap)




class Kawasaki_lx(QtWidgets.QWidget):

    def __init__(self, parent=None):
        QtWidgets.QWidget.__init__(self, parent)
        self.initTorch()
        self.scale = 1
        # self.width_pic = rsData.width
        # self.height_pic = rsData.height
        # self.radio = 1.33333
        self.initUI()
        self.extrinsic = None



    def initTorch(self):
        # rsData.torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        rsData.torch_device = torch.device("cpu")
        # "cuda:0" if torch.cuda.is_available() else
        # load model
        rsData.net = RevNet(baseNet='resnet34', pretrained=False)
        rsData.softmax = nn.Softmax(dim=1)

        # load model parameters
        # load model parameters
        modelPath = os.path.join(executable_path(),
                                 'models/snapshot-model_param_on_epoch_29_1T_gdd.pth')
        # modelPath = modelPath.replace('\\', '/')
        states = torch.load(modelPath, map_location=torch.device(rsData.torch_device))
        rsData.net.load_state_dict({k.replace('module.', ''): v for k, v in states['model'].items()})
        rsData.net = rsData.net.to(rsData.torch_device)
        rsData.softmax = rsData.softmax.to(rsData.torch_device)
        rsData.net.eval()

    def pcl(self):
        if not self.plot_container_pc:
            self.plot_container_pc = ThreeDSurfaceGraphWindowDlg()
        if self.plot_container_pc.isHidden():
            self.plot_container_pc.show()
            # self.plot_container_pc2.show()
        else:
            self.plot_container_pc.hide()
            # self.plot_container_pc2.hide()

    # def resizeEvent(self, event):
    #     w = self.gridWidget.width()/2
    #     h = self.gridWidget.height()/2
    #
    #
    #     newW = w
    #     newH = newW/self.radio
    #
    #     if newH>=h:
    #         pass
    #     else:
    #         newH = h
    #         newW = newH*self.radio
    #     self.scale = newW/rsData.width
    #     self.width_pic = newW
    #     self.height_pic = newH
    #
    #     self.update()


    def initUI(self):

        # self.virt()
        # self.control = Kawasaki_Control()
        self.label1 = ImageWidget()
        self.label2 = ImageWidget()
        ##self.label3 = ImageWidget()
        ##self.label4 = ImageWidget()

        # self.label1.setMinimumSize(QtCore.QSize(320,240))
        # self.label2.setMinimumSize(QtCore.QSize(320, 240))
        # self.label3.setMinimumSize(QtCore.QSize(320, 240))
        # self.label4.setMinimumSize(QtCore.QSize(320, 240))
        # self.label1.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored,QtWidgets.QSizePolicy.Policy.Ignored)
        # self.label2.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        # self.label3.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        # self.label4.setSizePolicy(QtWidgets.QSizePolicy.Policy.Ignored, QtWidgets.QSizePolicy.Policy.Ignored)
        # self.label1.resize(rsData.width, rsData.height)
        # self.label2.resize(rsData.width, rsData.height)
        # self.label3.resize(rsData.width, rsData.height)
        # self.label4.resize(rsData.width, rsData.height)

        self.plot_container_pc = None
        # self.plot_container_pc2 = None


        # self.plot_container_pc2 = ThreeDSurfaceGraphWindowDlg()


        self.hlayout = QtWidgets.QHBoxLayout()
        self.setLayout(self.hlayout)

        self.grid = QtWidgets.QGridLayout()
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.gridWidget = QtWidgets.QWidget()
        self.gridWidget.setLayout(self.grid)
        # self.grid.addWidget(self.control, 0, 0,1,2)

        self.grid.addWidget(self.label1, 0, 0)
        self.grid.addWidget(self.label2, 1, 0)
        ##self.grid.addWidget(self.label3, 1, 0)
        ##self.grid.addWidget(self.label4, 1, 1)
        # self.grid.addWidget(self.plot_container_pc1, 2, 0)
        # self.grid.addWidget(self.plot_container_pc2, 2, 1)



        self.hlayout.addWidget(self.gridWidget)
        # self.hlayout.addWidget(self.modeCntl)
        self.hlayout.setContentsMargins(0,0,0,0)

        self.setSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred,
                           QtWidgets.QSizePolicy.Policy.Preferred)

        tmp = cv2.cvtColor(rsData.depthImg.astype(np.uint8), cv2.COLOR_BGR2RGB).astype('uint8')
        qimg = QtGui.QImage(tmp, rsData.width,rsData.height, QtGui.QImage.Format_RGB888) ###########
        img = QtGui.QPixmap.fromImage(qimg)

        self.label1.setImage(qimg)
        self.label2.setImage(qimg)
        ##self.label3.setImage(qimg)
        ##self.label4.setImage(qimg)
        # img = img.scaled(self.width_pic,self.height_pic,QtCore.Qt.KeepAspectRatio)
        # self.label1.setPixmap(img)
        # self.label2.setPixmap(img)
        # self.label3.setPixmap(img)
        # self.label4.setPixmap(img)



    def paintEvent(self, event):
        self.showImage()

    def clearImageWidget(self):
        rsData.colorImg = np.zeros((rsData.height, rsData.width, 3), np.uint8)
        rsData.depthImg = np.zeros((rsData.height, rsData.width), np.uint16)
        rsData.inferImg = np.zeros((rsData.height, rsData.width, 3), np.uint8)
        rsData.clusters = np.zeros((rsData.height, rsData.width, 3), np.uint8)
        rsData.axisPoints = np.asarray([])
        rsData.axisPointsColor = np.asarray([])
        rsData.camPointsColor = np.asarray([])
        rsData.camPointsColor_clusters = np.asarray([])

        tmp3 = cv2.cvtColor(rsData.inferImg, cv2.COLOR_RGBA2RGB).astype('uint8')
        qImg3 = QtGui.QImage(tmp3, rsData.width,rsData.height, QtGui.QImage.Format_RGB888)
        tmp4 = cv2.cvtColor(rsData.clusters.astype(np.uint8), cv2.COLOR_RGBA2RGB).astype('uint8')
        qImg4 = QtGui.QImage(tmp4, rsData.width,rsData.height, QtGui.QImage.Format_RGB888)

        ## self.label3.setImage(qImg3)
        ##self.label4.setImage(qImg4)
        # img3 = QtGui.QPixmap.fromImage(qImg3)
        # img3 = img3.scaled(self.width_pic,self.height_pic)
        # self.label3.setPixmap(img3)

        # img4 = QtGui.QPixmap.fromImage(qImg4)
        # img4 = img4.scaled(self.width_pic, self.height_pic)
        # self.label4.setPixmap(img4)

        if self.plot_container_pc and not self.plot_container_pc.isHidden():
            self.plot_container_pc.plt1.draw_graph(rsData.pc1[0], rsData.pc1[1], rsData.pc1[2], rsData.camPointsColor,rsData.axisPointsColor)
            self.plot_container_pc.plt2.draw_graph(rsData.pc2[0], rsData.pc2[1], rsData.pc2[2], rsData.camPointsColor_clusters,rsData.axisPointsColor)


    def showRealTimeImage(self):
        tmp1 = cv2.cvtColor(rsData.color_streaming, cv2.COLOR_BGR2RGB).astype('uint8')
        qImg1 = QtGui.QImage(tmp1, rsData.width,rsData.height, QtGui.QImage.Format_RGB888)
        if rsData.mode == EMode.eRuntime and rsData.running == ERunning.eRunning:
            tmp2 = cv2.cvtColor(rsData.colorImg.astype(np.uint8), cv2.COLOR_RGBA2RGB).astype('uint8')
        else: #rsData.mode == EMode.eReview
            tmp2 = cv2.cvtColor(rsData.depthImg.astype(np.uint8), cv2.COLOR_RGBA2RGB).astype('uint8')
        qImg2 = QtGui.QImage(tmp2,  rsData.width,rsData.height, QtGui.QImage.Format_RGB888)

        self.label1.setImage(qImg1)
        self.label2.setImage(qImg2)
        # img1 = QtGui.QPixmap.fromImage(qImg1)
        # img1 = img1.scaled(self.width_pic, self.height_pic)
        # self.label1.setPixmap(img1)

        # img2 = QtGui.QPixmap.fromImage(qImg2)
        # img2 = img2.scaled(self.width_pic, self.height_pic)
        # self.label2.setPixmap(img2)

        tmp3 = cv2.cvtColor(rsData.inferImg, cv2.COLOR_RGBA2RGB).astype('uint8')
        qImg3 = QtGui.QImage(tmp3, rsData.width,rsData.height, QtGui.QImage.Format_RGB888)
        tmp4 = cv2.cvtColor(rsData.clusters.astype(np.uint8), cv2.COLOR_RGBA2RGB).astype('uint8')
        qImg4 = QtGui.QImage(tmp4, rsData.width,rsData.height, QtGui.QImage.Format_RGB888)

        ##self.label3.setImage(qImg3)
        ##self.label4.setImage(qImg4)
        # img3 = QtGui.QPixmap.fromImage(qImg3)
        # img3 = img3.scaled(self.width_pic, self.height_pic)
        # self.label3.setPixmap(img3)

        # img4 = QtGui.QPixmap.fromImage(qImg4)
        # img4 = img4.scaled(self.width_pic, self.height_pic)
        # self.label4.setPixmap(img4)

        if self.plot_container_pc and not self.plot_container_pc.isHidden():
            self.plot_container_pc.plt1.draw_graph(rsData.pc1[0], rsData.pc1[1], rsData.pc1[2], rsData.camPointsColor,rsData.axisPointsColor)
            self.plot_container_pc.plt2.draw_graph(rsData.pc2[0], rsData.pc2[1], rsData.pc2[2], rsData.camPointsColor_clusters,rsData.axisPointsColor)


    def showImage(self):
        if not rsData.b_stop_streaming:
            tmp1 = cv2.cvtColor(rsData.color_streaming, cv2.COLOR_BGR2RGB).astype('uint8')
        else:
            tmp1 = cv2.cvtColor(rsData.colorImg, cv2.COLOR_RGBA2RGB).astype('uint8')
        qImg1 = QtGui.QImage(tmp1, rsData.width,rsData.height, QtGui.QImage.Format_RGB888)
        if rsData.mode == EMode.eRuntime and rsData.running == ERunning.eRunning:
            tmp2 = cv2.cvtColor(rsData.colorImg.astype(np.uint8), cv2.COLOR_RGBA2RGB).astype('uint8')
        else:
            tmp2 = cv2.cvtColor(rsData.depthImg.astype(np.uint8), cv2.COLOR_RGBA2RGB).astype('uint8')
        qImg2 = QtGui.QImage(tmp2, rsData.width, rsData.height, QtGui.QImage.Format_RGB888)

        tmp3 = cv2.cvtColor(rsData.inferImg, cv2.COLOR_RGBA2RGB).astype('uint8')
        qImg3 = QtGui.QImage(tmp3, rsData.width,rsData.height, QtGui.QImage.Format_RGB888)
        tmp4 = cv2.cvtColor(rsData.clusters.astype(np.uint8), cv2.COLOR_RGBA2RGB).astype('uint8')
        qImg4 = QtGui.QImage(tmp4, rsData.width,rsData.height, QtGui.QImage.Format_RGB888)

        self.label1.setImage(qImg1)
        self.label2.setImage(qImg2)
        ##self.label3.setImage(qImg3)
        ##self.label4.setImage(qImg4)
        # img1 = QtGui.QPixmap.fromImage(qImg1)
        # img1 = img1.scaled(self.width_pic, self.height_pic)
        # self.label1.setPixmap(img1)

        # img2 = QtGui.QPixmap.fromImage(qImg2)
        # img2 = img2.scaled(self.width_pic, self.height_pic)
        # self.label2.setPixmap(img2)

        # img3 = QtGui.QPixmap.fromImage(qImg3)
        # img3 = img3.scaled(self.width_pic, self.height_pic)
        # self.label3.setPixmap(img3)

        # img4 = QtGui.QPixmap.fromImage(qImg4)
        # img4 = img4.scaled(self.width_pic, self.height_pic)
        # self.label4.setPixmap(img4)

        if self.plot_container_pc and not self.plot_container_pc.isHidden():
            self.plot_container_pc.plt1.draw_graph(rsData.pc1[0], rsData.pc1[1], rsData.pc1[2], rsData.camPointsColor,rsData.axisPointsColor)
            self.plot_container_pc.plt2.draw_graph(rsData.pc2[0], rsData.pc2[1], rsData.pc2[2], rsData.camPointsColor_clusters,rsData.axisPointsColor)


if __name__ == "__main__":
    rsData.torch_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load model
    rsData.net = RevNet(baseNet='resnet34', pretrained=False)
    rsData.softmax = nn.Softmax(dim=1)

    # load model parameters
    # load model parameters
    modelPath = os.path.join(executable_path(),
                             'models/snapshot-model_param_on_epoch_29_1T_gdd.pth')
    states = torch.load(modelPath, map_location=torch.device(rsData.torch_device))
    rsData.net.load_state_dict({k.replace('module.', ''): v for k, v in states['model'].items()})
    rsData.net = rsData.net.to(rsData.torch_device)
    rsData.softmax = rsData.softmax.to(rsData.torch_device)
    rsData.net.eval()
    app = QtWidgets.QApplication(sys.argv)
    kawasaki = Kawasaki_lx()
    kawasaki.show()

    sys.exit(app.exec_())
