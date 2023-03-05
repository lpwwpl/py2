'''
Michael Dawson-Haggerty

abb.py: contains classes and support functions which interact with an ABB Robot running our software stack (RAPID code module SERVER)


For functions which require targets (XYZ positions with quaternion orientation),
targets can be passed as [[XYZ], [Quats]] OR [XYZ, Quats]

'''

import socket
import json 
import time
import inspect
from threading import Thread
from collections import deque
import logging
from PySide2 import QtCore
import rsData
from concurrent.futures import ThreadPoolExecutor

from rsData import EStateMode
from PySide2 import QtConcurrent

class Job():
    def __init__(self,func,*params):
        self.func = func
        self.args = params

    def __call__(self):
        # param_count = len(self.args)
        self.func(self.args)
#QtCore.QThread
class Robot():
    def __init__(self, 
                 ip          = '127.0.0.1',
                 port_motion = 5000,
                 port_logger = 5001):
        # QtCore.QThread.__init__(self)
        self.delay   = .08
        self.working = True
        #connect_motion_thread = Thread(target = self.connect_motion,args = ((ip, port_logger))).start()
        self.connect_motion((ip, port_motion))
        # self.connect_logger((ip,port_logger))
        # log_thread = Thread(target = self.connect_logger,args = ((ip, port_logger))).start()

        self.queue = []
        self.sem = QtCore.QSemaphore(1)
        self.set_units('millimeters', 'degrees')
        rsData.state[1] = EStateMode.eNormal
        #self.set_tool()
        #self.set_workobject()
        # self.set_speed()
        # self.set_zone()

    # def run(self):
    #     while self.working == True:
    #         self.sem.acquire(1)
    #         while len(self.queue)>0:
    #             elem=self.queue.pop(0)
    #             elem()
    #         self.sem.release()
    #         time.sleep(0.5)

    def connect_motion(self, remote):        
        rsData.log.info('Attempting to connect to robot motion server at %s', str(remote))
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(0.5)
        self.sock.connect(remote)
        self.sock.settimeout(None)
        rsData.log.info('Connected to robot motion server at %s', str(remote))

    def connect_logger(self, remote, maxlen=10):
        self.pose   = deque(maxlen=maxlen)
        self.joints = deque(maxlen=maxlen)
        
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.connect(remote)
        s.setblocking(1)
        try:
            while True:
                #print(s.recv(4096))
                data = map(float, s.recv(4096).split())
                if int(data[1]) == 0:
                    rsData.robotPos = [data[2:5], data[5:]]
                elif int(data[1]) == 1: \
                    rsData.curJoints = [data[2:5], data[5:]]
        finally:
            s.shutdown(socket.SHUT_RDWR)
            rsData.state[1] = EStateMode.eWarning

    def set_units(self, linear, angular):
        units_l = {'millimeters': 1.0,
                   'meters'     : 1000.0,
                   'inches'     : 25.4}
        units_a = {'degrees' : 1.0,
                   'radians' : 57.2957795}
        self.scale_linear = units_l[linear]
        self.scale_angle  = units_a[angular]

    # def set_cartesian(self, pose):
    #     future = self.pool.submit(self.set_cartesian_,pose)
    #     # result = future.result()
    #     # return result

    def set_cartesian(self, pose):
        '''
        Executes a move immediately from the current pose,
        to 'pose', with units of millimeters.
        '''
        msg  = "01 " + self.format_pose(pose)
        # self.send_set(msg,False)
        return self.send(msg,False)

    def set_joints(self, joints):
        '''
        Executes a move immediately, from current joint angles,
        to 'joints', in degrees. 
        '''
        if len(joints) != 6: return False
        msg = "02 "
        for joint in joints: msg += format(joint*self.scale_angle, "+08.2f") + " " 
        msg += "#"
        # self.send_set(msg, False)
        return self.send(msg,False)

    def get_cartesian(self):
        '''
        Returns the current pose of the robot, in millimeters
        '''
        msg = "03 #"
        data = self.send(msg).split()
        r = [float(s) for s in data]
        return [r[2:5], r[5:9]]

    def get_move_complete(self):
        msg = "93 #"
        result = 0
        data = self.send(msg).split()
        data_len = len(data)
        if data_len>=3 and data.count(b"93")>=1:
            index = data.index(b"93")
            rsData.log.debug("get_move_complete,%s",data)
            if index >= 0 and data_len>=index+2+1 and data[index+1]==b"1":
                result = int(data[index+2])
        return result


    def get_joints(self):
        '''
        Returns the current angles of the robots joints, in degrees. 
        '''
        msg = "04 #"
        data = self.send(msg).split()
        return [float(s) / self.scale_angle for s in data[2:8]]

    def get_external_axis(self):
        '''
        If you have an external axis connected to your robot controller
        (such as a FlexLifter 600, google it), this returns the joint angles
        '''
        msg = "05 #"
        data = self.send(msg).split()
        return [float(s) for s in data[2:8]]
       
    def get_robotinfo(self):
        '''
        Returns a robot- unique string, with things such as the
        robot's model number. 
        Example output from and IRB 2400:
        ['24-53243', 'ROBOTWARE_5.12.1021.01', '2400/16 Type B']
        '''
        msg = "98 #"
        data = str(self.send(msg))[5:].split('*')
        rsData.log.debug('get_robotinfo result: %s', str(data))
        return data

    def set_tool(self, tool=[[0,0,85], [1,0,0,0]]):
        '''
        Sets the tool centerpoint (TCP) of the robot. 
        When you command a cartesian move, 
        it aligns the TCP frame with the requested frame.
        
        Offsets are from tool0, which is defined at the intersection of the
        tool flange center axis and the flange face.
        '''
        msg       = "06 " + self.format_pose(tool)    
        self.send(msg)
        self.tool = tool

    def load_json_tool(self, file_obj):
        if file_obj.__class__.__name__ == 'str':
            file_obj = open(filename, 'rb');
        tool = check_coordinates(json.load(file_obj))
        self.set_tool(tool)
        
    def get_tool(self): 
        rsData.log.debug('get_tool returning: %s', str(self.tool))
        return self.tool

    def set_workobject(self, work_obj=[[0,0,0],[1,0,0,0]]):
        '''
        The workobject is a local coordinate frame you can define on the robot,
        then subsequent cartesian moves will be in this coordinate frame. 
        '''
        msg = "07 " + self.format_pose(work_obj)   
        self.send(msg)

    def set_speed(self, speed=[100,50,50,50]):
        '''
        speed: [robot TCP linear speed (mm/s), TCP orientation speed (deg/s),
                external axis linear, external axis orientation]
        '''

        if len(speed) != 4: return False
        msg = "08 " 
        msg += format(speed[0], "+08.1f") + " " 
        msg += format(speed[1], "+08.2f") + " "  
        msg += format(speed[2], "+08.1f") + " " 
        msg += format(speed[3], "+08.2f") + " #"     
        self.send(msg)

    def set_zone(self, 
                 zone_key     = 'z1', 
                 point_motion = False, 
                 manual_zone  = []):
        zone_dict = {'z0'  : [.3,.3,.03], 
                    'z1'  : [1,1,.1], 
                    'z5'  : [5,8,.8], 
                    'z10' : [10,15,1.5], 
                    'z15' : [15,23,2.3], 
                    'z20' : [20,30,3], 
                    'z30' : [30,45,4.5], 
                    'z50' : [50,75,7.5], 
                    'z100': [100,150,15], 
                    'z200': [200,300,30]}
        '''
        Sets the motion zone of the robot. This can also be thought of as
        the flyby zone, AKA if the robot is going from point A -> B -> C,
        how close do we have to pass by B to get to C
        
        zone_key: uses values from RAPID handbook (stored here in zone_dict)
        with keys 'z*', you should probably use these

        point_motion: go to point exactly, and stop briefly before moving on

        manual_zone = [pzone_tcp, pzone_ori, zone_ori]
        pzone_tcp: mm, radius from goal where robot tool centerpoint 
                   is not rigidly constrained
        pzone_ori: mm, radius from goal where robot tool orientation 
                   is not rigidly constrained
        zone_ori: degrees, zone size for the tool reorientation
        '''

        if point_motion: 
            zone = [0,0,0]
        elif len(manual_zone) == 3: 
            zone = manual_zone
        elif zone_key in zone_dict.keys(): 
            zone = zone_dict[zone_key]
        else: return False
        
        msg = "09 " 
        msg += str(int(point_motion)) + " "
        msg += format(zone[0], "+08.4f") + " " 
        msg += format(zone[1], "+08.4f") + " " 
        msg += format(zone[2], "+08.4f") + " #" 
        self.send(msg)

    def buffer_add(self, pose):
        '''
        Appends single pose to the remote buffer
        Move will execute at current speed (which you can change between buffer_add calls)
        '''
        msg = "30 " + self.format_pose(pose) 
        self.send(msg)

    def buffer_set(self, pose_list):
        '''
        Adds every pose in pose_list to the remote buffer
        '''
        self.clear_buffer()
        for pose in pose_list: 
            self.buffer_add(pose)
        if self.buffer_len() == len(pose_list):
            rsData.log.debug('Successfully added %i poses to remote buffer',
                      len(pose_list))
            return True
        else:
            rsData.log.warn('Failed to add poses to remote buffer!')
            self.clear_buffer()
            return False

    def clear_buffer(self):
        msg = "31 #"
        data = self.send(msg)
        if self.buffer_len() != 0:
            rsData.log.warn('clear_buffer failed! buffer_len: %i', self.buffer_len())
            raise NameError('clear_buffer failed!')
        return data

    def buffer_len(self):
        '''
        Returns the length (number of poses stored) of the remote buffer
        '''
        msg = "32 #"
        data = self.send(msg).split()
        return int(float(data[2]))

    def buffer_execute(self):
        '''
        Immediately execute linear moves to every pose in the remote buffer.
        '''
        msg = "33 #"
        return self.send(msg)

    def set_external_axis(self, axis_unscaled=[-550,0,0,0,0,0]):
        if len(axis_values) != 6: return False
        msg = "34 "
        for axis in axis_values:
            msg += format(axis, "+08.2f") + " " 
        msg += "#"   
        return self.send(msg)

    def move_circular(self, pose_onarc, pose_end):
        '''
        Executes a movement in a circular path from current position, 
        through pose_onarc, to pose_end
        '''
        msg_0 = "35 " + self.format_pose(pose_onarc)  
        msg_1 = "36 " + self.format_pose(pose_end)

        data = self.send(msg_0).split()
        if data[1] != '1': 
            rsData.log.warn('move_circular incorrect response, bailing!')
            return False
        return self.send(msg_1)

    def set_gripper_on(self):
        msg = '94 ' + '#'
        return self.send(msg)

    def set_gripper_off(self):
        msg = '95 ' + '#'
        return self.send(msg)

    def read_dio(self,value,id):
        msg = '96 ' + value + ' ' + id + ' #'
        # return
        return self.send(msg)

    def set_dio(self, value, id):
        '''
        A function to set a physical DIO line on the robot.
        For this to work you're going to need to edit the RAPID function
        and fill in the DIO you want this to switch. 
        '''
        msg = '97 ' + value + ' ' + id + ' #'
        # return
        return self.send(msg)

    # def send_set(self,message, wait_for_response=True):
    #     job = Job(self.sendMsg,message,wait_for_response)
    #     self.sem.tryAcquire(1)
    #     self.queue.append(job)
    #     self.sem.release()
    #     print(len(self.queue))

    # def sendMsg(self,param):
    #     if isinstance(param,tuple) and len(param) == 2:
    #         for elem in param:
    #             self.send(param[0],param[1])
    #     else:
    #         self.send(param)

    def send(self,message, wait_for_response=True):
        '''
        Send a formatted message to the robot socket.
        if wait_for_response, we wait for the response and return it
        '''
        try:
            caller = inspect.stack()[1][3]
            # rsData.log.info('%-14s sending: %s', caller, message)
            self.sock.send(bytes(message,'utf-8'))
            time.sleep(self.delay)
            if not wait_for_response: return
            data = self.sock.recv(4096)
            # rsData.log.info('%-14s recieved: %s', caller, data)
            return data
        except Exception as e:
            rsData.state[1]=EStateMode.eErr_Critical
            self.sock.shutdown(socket.SHUT_RDWR)
            self.sock.close()
        return None
        
    def format_pose(self, pose):
        pose = check_coordinates(pose)
        msg  = ''
        for cartesian in pose[0]:
            msg += format(cartesian * self.scale_linear,  "+08.1f") + " " 
        for quaternion in pose[1]:
            msg += format(quaternion, "+08.5f") + " " 
        msg += "#" 
        return msg       
        
    def close(self):
        self.send("99 #", False)
        self.sock.shutdown(socket.SHUT_RDWR)
        self.sock.close()
        rsData.log.info('Disconnected from ABB robot.')

    def __enter__(self):
        return self
        
    def __exit__(self, type, value, traceback):
        self.close()

def check_coordinates(coordinates):
    if ((len(coordinates) == 2) and
        (len(coordinates[0]) == 3) and 
        (len(coordinates[1]) == 4)): 
        return coordinates
    elif (len(coordinates) == 7):
        return [coordinates[0:3], coordinates[3:7]]
    rsData.log.warn('Recieved malformed coordinate: %s', str(coordinates))
    raise NameError('Malformed coordinate!')


if __name__ == '__main__':
    formatter = logging.Formatter("[%(asctime)s] %(levelname)-7s (%(filename)s:%(lineno)3s) %(message)s", "%Y-%m-%d %H:%M:%S")
    handler_stream = logging.StreamHandler()
    handler_stream.setFormatter(formatter)
    handler_stream.setLevel(logging.DEBUG)
    log = logging.getLogger('abb')
    log.setLevel(logging.DEBUG)
    log.addHandler(handler_stream)
    r = Robot()
    # r.set_gripper_on()
    # r.set_gripper_off()
    # r.set_dio("do_21","1")
    r.read_dio("di_1","1")
    time.sleep(1)
    # r.set_dio('DO01')
    # while True:   
    #     pose = None
    #     if(r.pose.count()>0):
    #         pose = r.pose.pop()
    #     print(pose)
    #pose =
    #r.set_cartesian()
    # [28.78, -395.80, 388.629], [0.137, -0.027, 0.99, -0.017]
    #cartesian = [[19.78400213248231, -395.8032890167017, 388.6296361174512], [0.03616306185722351, -0.9973706007003784, -0.06096170097589493, 0.01509618107229471]]
    #xxxxxxxxxxxxxxxxxxxx cartesian = [[44.58661591970266, -379.2128474373975, 347.3478125700541], [0.04328731447458267, -0.9953938126564026, 0.02380471304059029, 0.0821625292301178]]
    #cartesian = [[44.60077369871478, -356.8234008223283, 389.5922902263236], [0.04451264813542366, -0.9987746477127075, 0.008489225059747696, 0.01989302597939968]]
    #cartesian = [[22.2055837423845, -367.6657275982245, 389.5748830663264], [-0.03923138231039047, 0.995405375957489, -0.03338910639286041, 0.0807100236415863]]
    #cartesian = [[68.92165801477529, -486.2578756020861, 389.5713521156477], [-0.004944915417581797, -0.9595786333084106, -0.2796183526515961, 0.03159059211611748]]
    #cartesian = [[87.68665499497214, -429.3362425881242, 361.7564646075469], [0.07170375436544418, -0.9944438934326172, -0.07045738399028778, 0.03123481757938862]]
    #cartesian = [[87.6941256249296, -415.0018815125193, 334.8238036016923], [0.1271899044513702, -0.9888269305229187, -0.06649995595216751, 0.04027154296636581]]
    #cartesian = [[87.7084609153225, -448.661347492412, 392.8498877227447], [-0.002661966485902667, -0.9820759296417236, -0.1858024150133133, 0.03157920390367508]]

    #cartesian = [[130.8713254306706, -356.4180250843793, 415.4987411658488], [0.1152927801012993, -0.9912218451499939, -0.01868686266243458, 0.06194836646318436]]
    #xxxxxxxxx cartesian = [[44.79907069273472, -366.2084801792037, 320.3411224127752], [0.08120923489332199, -0.9947656393051147, 0.05878718197345734, 0.01975796744227409]]
    #xxxxxxxxx cartesian = [[100.7241454909848, -323.230069456668, 320.3620969585789], [-0.2566156983375549, 0.9589929580688477, 0.1119959279894829, 0.04402045160531998]]
    #xxxxxxxxx cartesian = [[100.4735044938063, -299.7336402628107, 376.3814620332467], [-0.2576634287834167, 0.9529405236244202, 0.1488930284976959, 0.05783480033278465]]
    #xxxxxxxxx cartesian = [[42.26542441807317, -283.3228855483327, 376.3507139273154], [-0.1794711202383041, 0.9721313714981079, 0.1368243992328644, 0.06348059326410294]]
    #xxxxxxxxx cartesian = [[42.261397671941, -283.3328555479276, 397.4183770064311], [-0.2717323005199432, 0.9487152099609375, 0.151138037443161, 0.05708131939172745]]
    # cartesian = [[38.6086162787127, -341.127312091404, 397.395804230222], [-0.2182976305484772, 0.970672070980072, 0.01633049547672272, 0.09937518835067749]]



    #19.784002132482312, -395.8032890167017, 288.6296361174512, 0.0740686764948415, -3.115885908055233, -0.12304546049499021
    # cartesian = [[19.78400213248231, -395.8032890167017, 288.6296361174512], [0.03616306185722351, -0.9973706007003784, -0.06096170097589493, 0.01509618107229471]]
    # cartesian = [[19.78400213248231, -395.8032890167017, 288.6296361174512], [0.03616306185722351, -0.9973706007003784, -0.06096170097589493, 0.01509618107229471]]
    #r.get_cartesian()
    # r.set_cartesian(cartesian)


    # cartesian = r.get_cartesian()
    # r.get_joints()
    # print(cartesian)
    #joints = [5.41, -7.18, 5.62, 2.43, -3.5, -4.23]
    #r.set_joints(joints)

    
