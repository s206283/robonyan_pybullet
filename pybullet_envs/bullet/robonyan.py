import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import numpy as np
import copy
import math
import pybullet_data
import time
import random
import gym
from gym import spaces
from gym.utils import seeding
from pkg_resources import parse_version


class Robonyan:

    def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.maxVelocity = .35
        self.maxForce = 200.
        #self.fingerAForce = 2
        #self.fingerBForce = 2.5
        #self.fingerTipForce = 2
        self.useInverseKinematics = 1
        self.useSimulation = 1
        #self.useNullSpace = 21
        self.useNullSpace = 1
        self.useOrientation = 1
        self.proximity_L1 = 11
        self.proximity_L2 = 18
        self.proximity_L3 = 25
        self.proximity_R1 = 39
        self.proximity_R2 = 46
        self.proximity_R3 = 53
        self.proximity_list = np.array([self.proximity_L1, self.proximity_L2, self.proximity_L3,
                                    self.proximity_R1, self.proximity_R2, self.proximity_R3])
        self.force_L1 = 13
        self.force_L2 = 20
        self.force_L3 = 27
        self.force_R1 = 41
        self.force_R2 = 48
        self.force_R3 = 55

        self.L_EndEffectorIndex = 5
        self.R_EndEffectorIndex = 33

        self.kinect_rgb_width = 1920
        self.kinect_rgb_height = 1080
        self.kinect_d_width = 512
        self.kinect_d_height = 424

        self.handcamera_width = 640
        self.handcamera_height = 480


        #lower limits for null space
        #self.ll = [-2.70, -1.48, -2.96, -2.87, -2.00, -3.05]
        #upper limits for null space
        #self.ul = [2.70, -1.48, 0.87, 2.87, 2.00, 3.05]
        #joint ranges for null space
        #self.jr = [5.8, 2.96, 3.83, 5.8, 4, 6.1]
        #restposes for null space
        #self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66]
        #joint damping coefficents


        """
        self.jd = [
                    0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
                    0.00001, 0.00001, 0.00001, 0.00001
                    ]
        """
        #self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.reset()

    def rotate_x(self, x):
        r = np.float32(x)
        c = np.cos(r)
        s = np.sin(r)
        # 回転行列Rx
        Rx = np.matrix([[1,0,0],
            [0,c,-s],
            [0,s,c]])

        return Rx

    def rotate_y(self, y):
        r = np.float32(y)
        c = np.cos(r)
        s = np.sin(r)
        # 回転行列Ry
        Ry = np.matrix([[c,0,s],
            [0,1,0],
            [-s,0,c]])

        return Ry

    def rotate_z(self, z):
        r = np.float32(z)
        c = np.cos(r)
        s = np.sin(r)
        # 回転行列Rx
        Rz = np.matrix([[c,-s,0],
            [s,c,0],
            [0,0,1]])

        return Rz


    def getJointRanges(self, bodyId, includeFixed=False):

        """
        Parameters
        ----------
        bodyId : int
        includeFixed : bool
        Returns
        -------
        lowerLimits : [ float ] * numDofs
        upperLimits : [ float ] * numDofs
        jointRanges : [ float ] * numDofs
        restPoses : [ float ] * numDofs
        """

        lowerLimits, upperLimits, jointRanges, restPoses = [], [], [], []

        numJoints = p.getNumJoints(bodyId)
        #print(numJoints)

        for i in range(numJoints):
            jointInfo = p.getJointInfo(bodyId, i)

            if includeFixed or jointInfo[3] > -1:

                ll, ul = jointInfo[8:10]
                jr = ul - ll

                # For simplicity, assume resting state == initial state
                rp = p.getJointState(bodyId, i)[0]

                lowerLimits.append(ll)
                upperLimits.append(ul)
                jointRanges.append(jr)
                restPoses.append(rp)

        return lowerLimits, upperLimits, jointRanges, restPoses


    def reset(self):
        objects = p.loadSDF(os.path.join(self.urdfRootPath, "robonyan_gripper/robonyan_test.sdf"))
        self.robonyanUid = objects[0]
        #for i in range (p.getNumJoints(self.kukaUid)):
        #  print(p.getJointInfo(self.kukaUid,i))
        p.resetBasePositionAndOrientation(self.robonyanUid, [-0.100000, 0.000000, 0.30000],
                                                [0.000000, 0.000000, 0.000000, 1.000000])

        # 各ジョイントの初期角度
        """
        self.jointPositions = [
            0.006418, 0.413184, -0.011401, -1.589317, 0.005379, 1.137684, -0.006539, 0.000048,
            -0.299912, 0.000000, -0.000043, 0.299960, 0.000000, -0.000200
            ]
        """
        #for jointIndex in range(numJoints):
        #p.resetJointState(robonyanUid, jointIndex, 0)
        #p.setJointMotorControl2(robonyanUid,
                                #jointIndex,
                                #p.POSITION_CONTROL,
                                #targetPosition=0,
                                #force=maxForce)
        self.numJoints = p.getNumJoints(self.robonyanUid)
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.robonyanUid, jointIndex, 0)
            p.setJointMotorControl2(self.robonyanUid,
                                    jointIndex,
                                    p.POSITION_CONTROL,
                                    targetPosition=0,
                                    force=self.maxForce)

        self.ll, self.ul, self.jr, self.rp = self.getJointRanges(self.robonyanUid, includeFixed=False)

        self.jd = [0.1] * self.numJoints

        """
        self.trayUid = p.loadURDF(os.path.join(self.urdfRootPath, "tray/tray.urdf"), 0.640000,
                                    0.075000, -0.190000, 0.000000, 0.000000, 1.000000, 0.000000)
        self.endEffectorPos = [0.537, 0.0, 0.5]
        self.endEffectorAngle = 0
        """

        # endEffectorPosを3指の中点に変更するべき
        L_state = p.getLinkState(self.robonyanUid, self.L_EndEffectorIndex)
        R_state = p.getLinkState(self.robonyanUid, self.R_EndEffectorIndex)
        self.L_EndEffectorPos = list(L_state[0])
        self.R_EndEffectorPos = list(R_state[0])
        #print(self.L_EndEffectorPos)
        #print(self.R_EndEffectorPos)

        """
        # 各ジョイントのmotorNAmes, motorIndicesのリスト作成
        self.leftmotorNames = []
        self.leftmotorIndices = []
        # lefthandのアームジョイントは0～5
        self.leftnumjoints = [i for i in range(6)]


        #for i in range(self.leftnumjoints):
        for i in range(6):
            jointInfo = p.getJointInfo(self.robonyanUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                #print("motorname")
                #print(jointInfo[1])
                self.leftmotorNames.append(str(jointInfo[1]))
                self.leftmotorIndices.append(i)

        # 各ジョイントのmotorNAmes, motorIndicesのリスト作成
        self.rightmotorNames = []
        self.rightmotorIndices = []
        # Righthandのアームジョイントは28～33
        self.rightnumjoints = [i for i in range(28, 34)]

        #for i in range(self.rightnumjoints):
        for i in range(28, 34):
            jointInfo = p.getJointInfo(self.robonyanUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                #print("motorname")
                #print(jointInfo[1])
                self.rightmotorNames.append(str(jointInfo[1]))
                self.rightmotorIndices.append(i)
        """
        self.motorNames = []
        self.motorIndices = []

        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.robonyanUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                #print("motorname")
                #print(jointInfo[1])
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)

        self.ll, self.ul, self.jr, self.rp = self.getJointRanges(self.robonyanUid, includeFixed=False)

        self.jd = [0.1] * len(self.motorIndices)
        #print(len(self.motorIndices))

    def getActionDimension(self):
        if (self.useInverseKinematics):
            #return len(self.rightmotorIndices)
            return (self.motorIndices)
        return 6  #position x,y,z and roll/pitch/yaw euler angles of end effector

    def getObservationDimension(self):
        return len(self.getObservation())


    # RobonyanGymEnvに作成するため、不要（画像などセンサを用いない場合を作成し、比較してもいいかも）
    def getObservation(self):
        #observation = []
        # 両手の手先の状態を取得

        # kinectv2のカメラ座標取得
        kinect = p.getLinkState(self.robonyanUid, 56)
        kinectPos,kinectOrn = kinect[0], kinect[1]
        kinectPos = list(kinectPos)
        kinectPos[0] += 0.2
        kinectPos[1] += 0.002
        kinectPos[2] += 0.05
        #kinectPos = [-0.112 + 0.03153696, 0, 1.304 + 0.03153696]
        kinectPos = tuple(kinectPos)
        #kinectEuler = p.getEulerFromQuaternion(kinectOrn)
        #kinectYaw = kinectEuler[2]*360/(2.*math.pi)-90

        camInfo = p.getDebugVisualizerCamera()

        kinectMat = p.getMatrixFromQuaternion(kinectOrn)
        upVector = [0,0,1]
        forwardVec = [kinectMat[0],kinectMat[3],kinectMat[6]]
        #sideVec =  [camMat[1],camMat[4],camMat[7]]
        kinectUpVec =  [kinectMat[2],kinectMat[5],kinectMat[8]]
        kinectTarget = [kinectPos[0]+forwardVec[0]*100,kinectPos[1]+forwardVec[1]*100,kinectPos[2]+forwardVec[2]*100]
        kinectUpTarget = [kinectPos[0]+kinectUpVec[0],kinectPos[1]+kinectUpVec[1],kinectPos[2]+kinectUpVec[2]]
        kinectviewMat = p.computeViewMatrix(kinectPos, kinectTarget, kinectUpVec)
        kinectprojMat = camInfo[3]
        #p.getCameraImage(320,200,viewMatrix=viewMat,projectionMatrix=projMat, flags=p.ER_NO_SEGMENTATION_MASK, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        #p.getCameraImage(width,height,viewMatrix=kinectviewMat,projectionMatrix=kinectprojMat, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        kinect_img_arr = p.getCameraImage(width=self.kinect_rgb_width,
                                          height=self.kinect_rgb_height,
                                          viewMatrix=kinectviewMat,
                                          projectionMatrix=kinectprojMat)
        kinect_rgb = kinect_img_arr[2]
        kinect_np_img_arr = np.reshape(kinect_rgb, (self.kinect_rgb_height, self.kinect_rgb_width, 4))

        self._kinect_observation = kinect_np_img_arr


        return self._kinect_observation

    # 両手の行動（はじめは片手のみ制御）
    def reachAction(self, actions):

        #print ("self.numJoints")
        #print (self.numJoints)
        self.max_pos = 0.02
        self.max_rot = 0.1745
        if (self.useInverseKinematics):
            #actions = np.array(actions)
            assert actions.shape == (6,)
            actions = actions.copy()
            #print(actions)
            #print(type(actions))
            pos_ctrl, rot_ctrl = actions[:3], actions[3:]
            #print(pos_ctrl)
            #print(rot_ctrl)
            #print(type(pos_ctrl))
            #print(type(rot_ctrl))
            pos_ctrl = np.clip(pos_ctrl, -self.max_pos, self.max_pos)
            rot_ctrl = np.clip(rot_ctrl, -self.max_rot, self.max_rot)
            pos_ctrl = np.array([pos_ctrl])
            rot_ctrl = np.array([rot_ctrl])
            rot_ctrl = rot_ctrl[0]
            #print(pos_ctrl)
            #print(rot_ctrl)
            #print(type(pos_ctrl))
            #print(type(rot_ctrl))
            dx = pos_ctrl[0][0]
            dy = pos_ctrl[0][1]
            dz = pos_ctrl[0][2]
            #da = motorCommands[3]
            #droll = rot_ctrl[0]
            #dpitch = rot_ctrl[1]
            #dyaw = rot_ctrl[2]
            #fingerAngle = motorCommands[4]

            # 両手の手先の状態を取得
            L_state = p.getLinkState(self.robonyanUid, self.L_EndEffectorIndex)
            R_state = p.getLinkState(self.robonyanUid, self.R_EndEffectorIndex)
            L_actualEndEffectorPos = L_state[0]
            R_actualEndEffectorPos = R_state[0]
            #print(R_actualEndEffectorPos)
            #print("pos[2] (getLinkState(kukaEndEffectorIndex)")
            #print(actualEndEffectorPos[2])

            #print(self.L_EndEffectorPos)
            #print(self.R_EndEffectorPos)
            #print(type(self.L_EndEffectorPos))

            self.L_EndEffectorPos[0] = self.L_EndEffectorPos[0] + dx
            if (self.L_EndEffectorPos[0] > 0.95):
                self.L_EndEffectorPos[0] = 0.95
            if (self.L_EndEffectorPos[0] < 0.58):
                self.L_EndEffectorPos[0] = 0.58
            self.L_EndEffectorPos[1] = self.L_EndEffectorPos[1] + dy
            if (self.L_EndEffectorPos[1] < 0.0):
                self.L_EndEffectorPos[1] = 0.0
            if (self.L_EndEffectorPos[1] > 0.65):
                self.L_EndEffectorPos[1] = 0.65
            #print ("self.endEffectorPos[2]")
            #print (self.endEffectorPos[2])
            #print("actualEndEffectorPos[2]")
            #print(actualEndEffectorPos[2])
            #if (dz<0 or actualEndEffectorPos[2]<0.5):
            self.L_EndEffectorPos[2] = self.L_EndEffectorPos[2] + dz
            if (self.L_EndEffectorPos[2] > 0.9):
                self.L_EndEffectorPos[2] = 0.9
            if (self.L_EndEffectorPos[2] < 0.68):
                self.L_EndEffectorPos[2] = 0.68



            self.R_EndEffectorPos[0] = self.R_EndEffectorPos[0] + dx
            if (self.R_EndEffectorPos[0] > 0.95):
                self.R_EndEffectorPos[0] = 0.95
            if (self.R_EndEffectorPos[0] < 0.58):
                self.R_EndEffectorPos[0] = 0.58
            self.R_EndEffectorPos[1] = self.R_EndEffectorPos[1] + dy
            if (self.R_EndEffectorPos[1] < -0.65):
                self.R_EndEffectorPos[1] = -0.65
            if (self.R_EndEffectorPos[1] > 0.0):
                self.R_EndEffectorPos[1] = 0.0
            #print ("self.endEffectorPos[2]")
            #print (self.endEffectorPos[2])
            #print("actualEndEffectorPos[2]")
            #print(actualEndEffectorPos[2])
            #if (dz<0 or actualEndEffectorPos[2]<0.5):
            self.R_EndEffectorPos[2] = self.R_EndEffectorPos[2] + dz
            if (self.R_EndEffectorPos[2] > 0.9):
                self.R_EndEffectorPos[2] = 0.9
            if (self.R_EndEffectorPos[2] < 0.68):
                self.R_EndEffectorPos[2] = 0.68

            #self.endEffectorAngle = self.endEffectorAngle + da
            L_pos = self.L_EndEffectorPos
            R_pos = self.R_EndEffectorPos
            #print(R_pos)
            #orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
            orn = p.getQuaternionFromEuler(rot_ctrl)
            if (self.useNullSpace == 1):
                if (self.useOrientation == 1):
                    #The base and fixed joints are skipped
                    jointPoses = p.calculateInverseKinematics(self.robonyanUid, self.R_EndEffectorIndex, R_pos,
                                                            orn, self.ll, self.ul, self.jr, self.rp)
                else:
                    jointPoses = p.calculateInverseKinematics(self.robonyanUid,
                                                              self.R_EndEffectorIndex,
                                                              R_pos,
                                                              lowerLimits=self.ll,
                                                              upperLimits=self.ul,
                                                              jointRanges=self.jr,
                                                              restPoses=self.rp)
            else:
                if (self.useOrientation == 1):
                    jointPoses = p.calculateInverseKinematics(self.robonyanUid,
                                                              self.R_EndEffectorIndex,
                                                              R_pos,
                                                              orn,
                                                              jointDamping=self.jd)
                else:
                    jointPoses = p.calculateInverseKinematics(self.robonyanUid, self.R_EndEffectorIndex, R_pos)

            #print("jointPoses")
            #print(len(jointPoses))

            #print("self.R_EndEffectorIndex")
            #print(self.R_EndEffectorIndex)

            if (self.useSimulation):
                #for i in range(self.R_EndEffectorIndex + 1):
                    #p.resetJointState(self.robonyanUid, i, jointPoses[i])
                j = 0
                for i in range(self.numJoints):
                    jointInfo = p.getJointInfo(self.robonyanUid, i)
                    #type = jointInfo[2]
                    qIndex = jointInfo[3]
                    #if type == 'revolute':
                    if qIndex > -1:
                        p.resetJointState(self.robonyanUid,i,jointPoses[j])
                        j += 1
                #print(j)

                #for i in range(28, 34):
                #for i in range(self.R_EndEffectorIndex + 1):
                j = 0
                for i in range(self.numJoints):
                    #print(i)
                    jointInfo = p.getJointInfo(self.robonyanUid, i)
                    #type = jointInfo[2]
                    qIndex = jointInfo[3]
                    #if type == 'revolute':
                    if qIndex > -1:
                        p.setJointMotorControl2(bodyUniqueId=self.robonyanUid,
                                                jointIndex=i,
                                                controlMode=p.POSITION_CONTROL,
                                                targetPosition=jointPoses[j],
                                                targetVelocity=0,
                                                force=self.maxForce,
                                                maxVelocity=self.maxVelocity,
                                                positionGain=0.3,
                                                velocityGain=1)
                        j += 1
            else:
                #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
                for i in range(self.numJoints):
                    p.resetJointState(self.robonyanUid, i, jointPoses[i])
            #fingers
            """
            p.setJointMotorControl2(self.kukaUid,
                                    7,
                                    p.POSITION_CONTROL,
                                    targetPosition=self.endEffectorAngle,
                                    force=self.maxForce)
            p.setJointMotorControl2(self.kukaUid,
                                    8,
                                    p.POSITION_CONTROL,
                                    targetPosition=-fingerAngle,
                                    force=self.fingerAForce)
            p.setJointMotorControl2(self.kukaUid,
                                    11,
                                    p.POSITION_CONTROL,
                                    targetPosition=fingerAngle,
                                    force=self.fingerBForce)

            p.setJointMotorControl2(self.kukaUid,
                                    10,
                                    p.POSITION_CONTROL,
                                    targetPosition=0,
                                    force=self.fingerTipForce)
            p.setJointMotorControl2(self.kukaUid,
                                    13,
                                    p.POSITION_CONTROL,
                                    targetPosition=0,
                                    force=self.fingerTipForce)
            """

        else:
            for action in range(len(actions)):
                motor = self.motorIndices[action]
                p.setJointMotorControl2(self.robonyanUid,
                                        motor,
                                        p.POSITION_CONTROL,
                                        targetPosition=actions[action],
                                        force=self.maxForce)
