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
        self.fingerForce = 2.5
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

        self.Lfinger_list = np.array([8,9,10,12,15,16,17,19,22,23,24,26])
        self.Rfinger_list = np.array([36,37,38,40,43,44,45,47,50,51,52,54])

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


        self.numJoints = p.getNumJoints(self.robonyanUid)
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.robonyanUid, jointIndex, 0)
            p.setJointMotorControl2(self.robonyanUid,
                                    jointIndex,
                                    p.POSITION_CONTROL,
                                    targetPosition=0,
                                    force=self.maxForce)

        L_state = p.getLinkState(self.robonyanUid, self.L_EndEffectorIndex)
        R_state = p.getLinkState(self.robonyanUid, self.R_EndEffectorIndex)
        self.L_EndEffectorPos = list(L_state[0])
        self.R_EndEffectorPos = list(R_state[0])

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

    def getObservation(self):

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

    def getPregraspPosition(self):
        L_proximity_1 = p.getLinkState(self.robonyanUid, self.proximity_L1)
        L_proximity_2 = p.getLinkState(self.robonyanUid, self.proximity_L2)
        L_proximity_3 = p.getLinkState(self.robonyanUid, self.proximity_L3)

        R_proximity_1 = p.getLinkState(self.robonyanUid, self.proximity_R1)
        R_proximity_2 = p.getLinkState(self.robonyanUid, self.proximity_R2)
        R_proximity_3 = p.getLinkState(self.robonyanUid, self.proximity_R3)

        L1_prox_pos = list(L_proximity_1[0])
        L2_prox_pos = list(L_proximity_2[0])
        L3_prox_pos = list(L_proximity_3[0])

        L_x = (L1_prox_pos[0] + L2_prox_pos[0] + L3_prox_pos[0]) / 3
        L_y = (L1_prox_pos[1] + L2_prox_pos[1] + L3_prox_pos[1]) / 3
        L_z = (L1_prox_pos[2] + L2_prox_pos[2] + L3_prox_pos[2]) / 3

        R1_prox_pos = list(R_proximity_1[0])
        R2_prox_pos = list(R_proximity_2[0])
        R3_prox_pos = list(R_proximity_3[0])

        R_x = (R1_prox_pos[0] + R2_prox_pos[0] + R3_prox_pos[0]) / 3
        R_y = (R1_prox_pos[1] + R2_prox_pos[1] + R3_prox_pos[1]) / 3
        R_z = (R1_prox_pos[2] + R2_prox_pos[2] + R3_prox_pos[2]) / 3

        self.L_pregrasp = np.array([L_x, L_y, L_z])
        self.R_pregrasp = np.array([R_x, R_y, R_z])

        return self.L_pregrasp, self.R_pregrasp

    def JointControl(self, jointPoses):
        j = 0
        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.robonyanUid, i)
            #type = jointInfo[2]
            qIndex = jointInfo[3]
            #if type == 'revolute':
            if qIndex > -1:
                p.resetJointState(self.robonyanUid,i,jointPoses[j])
                j += 1

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

    def resetHandPosition(self, R_position, L_position):
        R_pos, R_orn = R_position[:3], R_position[3:]
        L_pos, L_orn = L_position[:3], L_position[3:]

        #orn = [1.5708, 0, 0]
        R_orn = p.getQuaternionFromEuler(R_orn)
        L_orn = p.getQuaternionFromEuler(L_orn)

        jointPoses = p.calculateInverseKinematics(self.robonyanUid, self.L_EndEffectorIndex, L_pos,
                                                L_orn, self.ll, self.ul, self.jr, self.rp)
        self.JointControl(jointPoses)

        R_state = p.getLinkState(self.robonyanUid, self.R_EndEffectorIndex)
        R_EndEffectorPos = R_state[0]

        jointPoses = p.calculateInverseKinematics(self.robonyanUid, self.R_EndEffectorIndex, R_EndEffectorPos,
                                                R_orn, self.ll, self.ul, self.jr, self.rp)
        self.JointControl(jointPoses)

        pos_vec = np.array(R_pos)
        L_pregrasp, R_pregrasp =  self.getPregraspPosition()
        R_d = pos_vec - R_pregrasp
        R_state = p.getLinkState(self.robonyanUid, self.R_EndEffectorIndex)

        R_EndEffectorPos = R_state[0]
        pos = R_EndEffectorPos + R_d
        pos = pos.tolist()

        jointPoses = p.calculateInverseKinematics(self.robonyanUid, self.R_EndEffectorIndex, pos,
                                                R_orn, self.ll, self.ul, self.jr, self.rp)

        self.JointControl(jointPoses)

    def applyAction(self, actions):

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


        else:
            for action in range(len(actions)):
                motor = self.motorIndices[action]
                p.setJointMotorControl2(self.robonyanUid,
                                        motor,
                                        p.POSITION_CONTROL,
                                        targetPosition=actions[action],
                                        force=self.maxForce)

    def graspAction(self, actions):
        assert actions.shape == (12,)
        actions = actions.copy()

        for action in range(len(actions)):
            motor = self.Rfinger_list[action]
            p.resetJointState(self.robonyanUid,motor,actions[action])

        for action in range(len(actions)):
            motor = self.Rfinger_list[action]
            p.setJointMotorControl2(self.robonyanUid,
                                    motor,
                                    p.POSITION_CONTROL,
                                    targetPosition=actions[action],
                                    force=self.fingerForce)
