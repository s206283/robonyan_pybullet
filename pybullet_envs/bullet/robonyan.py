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


class Kuka:
    metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

    def rotate_x(x):
        r = np.float32(x)
        c = np.cos(r)
        s = np.sin(r)
        # 回転行列Rx
        Rx = np.matrix([[1,0,0],
            [0,c,-s],
            [0,s,c]])

        return Rx

    def rotate_y(y):
        r = np.float32(y)
        c = np.cos(r)
        s = np.sin(r)
        # 回転行列Ry
        Ry = np.matrix([[c,0,s],
            [0,1,0],
            [-s,0,c]])

        return Ry

    def rotate_z(z):
        r = np.float32(z)
        c = np.cos(r)
        s = np.sin(r)
        # 回転行列Rx
        Rz = np.matrix([[c,-s,0],
            [s,c,0],
            [0,0,1]])

        return Rz

  def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self.maxVelocity = .35
    self.maxForce = 200.
    self.fingerAForce = 2
    self.fingerBForce = 2.5
    self.fingerTipForce = 2
    self.useInverseKinematics = 1
    self.useSimulation = 1
    self.useNullSpace = 21
    self.useOrientation = 1

    self.proximity_L1 = 11
    self.proximity_L2 = 18
    self.proximity_L3 = 25
    self.proximity_R1 = 39
    self.proximity_R2 = 46
    self.proximity_R3 = 53
    self.proximity_list = np.array([self.proximity_L1, self.proximity_L2, self.proximity_L3, self.proximity_R1, self.proximity_R2, self.proximity_R3])
    self.force_L1 = 13
    self.force_L2 = 20
    self.force_L3 = 27
    self.force_R1 = 41
    self.force_R2 = 48
    self.force_R3 = 55

    self.L_EndEffectorIndex = 5

    self.kinect_rgb_width = 1920
    self.kinect_rgb_height = 1080
    self.kinect_d_width = 512
    self.kinect_d_height = 424

    self.handcamera_width = 640
    self.handcamera_height = 480


    #lower limits for null space
    self.ll = [-2.70, -1.48, -2.96, -2.87, -2.00, -3.05]
    #upper limits for null space
    self.ul = [2.70, -1.48, 0.87, 2.87, 2.00, 3.05]
    #joint ranges for null space
    self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
    #restposes for null space
    self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
    #joint damping coefficents
    self.jd = [
        0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001,
        0.00001, 0.00001, 0.00001, 0.00001
    ]
    self.reset()

  def reset(self):
    objects = p.loadSDF(os.path.join(self.urdfRootPath, "robonyan_gripper/robonyan_test.sdf"))
    self.robonyanUid = objects[0]
    #for i in range (p.getNumJoints(self.kukaUid)):
    #  print(p.getJointInfo(self.kukaUid,i))
    p.resetBasePositionAndOrientation(robonyanUid, [-0.100000, 0.000000, 0.30000],
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

    """
    self.trayUid = p.loadURDF(os.path.join(self.urdfRootPath, "tray/tray.urdf"), 0.640000,
                              0.075000, -0.190000, 0.000000, 0.000000, 1.000000, 0.000000)
    self.endEffectorPos = [0.537, 0.0, 0.5]
    self.endEffectorAngle = 0
    """

    # 各ジョイントのmotorNAmes, motorIndicesのリスト作成
    self.leftmotorNames = []
    self.leftmotorIndices = []
    # lefthandのアームジョイントは0～5
    self.leftnumjoints = 5

    for i in range(self.leftnumjoints):
      jointInfo = p.getJointInfo(self.robonyanUid, i)
      qIndex = jointInfo[3]
      if qIndex > -1:
        #print("motorname")
        #print(jointInfo[1])
        self.leftmotorNames.append(str(jointInfo[1]))
        self.leftmotorIndices.append(i)

  def getActionDimension(self):
    if (self.useInverseKinematics):
      return len(self.leftmotorIndices)
    return 6  #position x,y,z and roll/pitch/yaw euler angles of end effector

  def getObservationDimension(self):
    return len(self.getObservation())

  def getObservation(self):
    observation = []
    # 両手の手先の状態を取得

    # kinectv2のカメラ座標取得
    kinectPos,kinectOrn = kinect[0], kinect[1]
    kinectPos = list(kinectPos)
    kinectPos[0] += 0.2
    kinectPos[1] += 0.002
    kinectPos[2] += 0.05
    #kinectPos = [-0.112 + 0.03153696, 0, 1.304 + 0.03153696]
    kinectPos = tuple(kinectPos)
    #kinectEuler = p.getEulerFromQuaternion(kinectOrn)
    #kinectYaw = kinectEuler[2]*360/(2.*math.pi)-90

    L_hand = p.getLinkState(robonyanUid, 6)

    R_hand = p.getLinkState(robonyanUid, 34)


    # L_handのカメラ座標取得
    L_handPos,L_handOrn = L_hand[0], L_hand[1]
    L_handEuler = p.getEulerFromQuaternion(L_handOrn)
    L_handYaw = L_handEuler[2]*360/(2.*math.pi)-90

    L_camera = np.array([0.01, 0, 0])
    L_handOrn = list(L_handOrn)
    # 回転行列の生成
    Rx = rotate_x(L_handOrn[0])
    Ry = rotate_y(L_handOrn[1])
    Rz = rotate_z(L_handOrn[2])
    # 回転後のベクトルを計算
    R1 = np.dot(Ry,Rx)
    R2 = np.dot(R1,Rz)
    R3 = np.dot(R2,L_camera)
    # 途中経過
    R4 = np.dot(Rz,L_camera)
    Ra = np.dot(Rx,Rz)
    R5 = np.dot(Ra,L_camera)
    # 整数型に変換
    b = np.array(R3,dtype=np.float32)
    L_handPos = np.array(L_handPos) + b
    L_handPos = L_handPos.tolist()
    L_handPos = L_handPos[0]
    #print(R_handPos)
    L_handPos = tuple(L_handPos)
    #print(R_handPos)
    R4 = np.array(R4,dtype=np.float32)
    R5 = np.array(R5,dtype=np.float32)

    L_handOrn = tuple(L_handOrn)

    # R_handのカメラ座標取得
    R_handPos,R_handOrn = R_hand[0], R_hand[1]
    R_handEuler = p.getEulerFromQuaternion(R_handOrn)
    R_handYaw = R_handEuler[2]*360/(2.*math.pi)-90


    R_camera = np.array([0.01, 0, 0])
    R_handOrn = list(R_handOrn)
    # 回転行列の生成
    Rx = rotate_x(R_handOrn[0])
    Ry = rotate_y(R_handOrn[1])
    Rz = rotate_z(R_handOrn[2])
    # 回転後のベクトルを計算
    R1 = np.dot(Ry,Rx)
    R2 = np.dot(R1,Rz)
    R3 = np.dot(R2,R_camera)
    # 途中経過
    R4 = np.dot(Rz,R_camera)
    Ra = np.dot(Rx,Rz)
    R5 = np.dot(Ra,R_camera)
    # 整数型に変換
    b = np.array(R3,dtype=np.float32)
    R_handPos = np.array(R_handPos) + b
    R_handPos = R_handPos.tolist()
    R_handPos = R_handPos[0]
    #print(R_handPos)
    R_handPos = tuple(R_handPos)
    #print(R_handPos)
    R4 = np.array(R4,dtype=np.float32)
    R5 = np.array(R5,dtype=np.float32)

    R_handOrn = tuple(R_handOrn)
    #print(R_handOrn)

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

    L_handMat = p.getMatrixFromQuaternion(L_handOrn)
    upVector = [0,0,1]
    forwardVec = [L_handMat[0],L_handMat[3],L_handMat[6]]
    #sideVec =  [camMat[1],camMat[4],camMat[7]]
    L_handUpVec =  [L_handMat[2],L_handMat[5],L_handMat[8]]
    L_handTarget = [L_handPos[0]+forwardVec[0]*100,L_handPos[1]+forwardVec[1]*100,L_handPos[2]+forwardVec[2]*100]
    L_handUpTarget = [L_handPos[0]+L_handUpVec[0],L_handPos[1]+L_handUpVec[1],L_handPos[2]+L_handUpVec[2]]
    L_handviewMat = p.computeViewMatrix(L_handPos, L_handTarget, L_handUpVec)
    L_handprojMat = camInfo[3]
    #p.getCameraImage(320,200,viewMatrix=viewMat,projectionMatrix=projMat, flags=p.ER_NO_SEGMENTATION_MASK, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    L_hand_img_arr = p.getCameraImage(width=self.handcamera_width,
                                    height=self.handcamera_height,
                                    viewMatrix=L_handviewMat,
                                    projectionMatrix=L_handprojMat)
    L_hand_rgb = L_hand_img_arr[2]
    L_hand_np_img_arr = np.reshape(L_hand_rgb, (self.hand_camera_height, self.hand_camera_width, 4))

    #R_hand視点
    R_handMat = p.getMatrixFromQuaternion(R_handOrn)
    upVector = [0,0,1]
    forwardVec = [R_handMat[0],R_handMat[3],R_handMat[6]]
    #sideVec =  [camMat[1],camMat[4],camMat[7]]
    R_handUpVec =  [R_handMat[2],R_handMat[5],R_handMat[8]]
    R_handTarget = [R_handPos[0]+forwardVec[0]*100,R_handPos[1]+forwardVec[1]*100,R_handPos[2]+forwardVec[2]*100]
    R_handUpTarget = [R_handPos[0]+R_handUpVec[0],R_handPos[1]+R_handUpVec[1],R_handPos[2]+R_handUpVec[2]]
    R_handviewMat = p.computeViewMatrix(R_handPos, R_handTarget, R_handUpVec)
    R_handprojMat = camInfo[3]
    #p.getCameraImage(320,200,viewMatrix=viewMat,projectionMatrix=projMat, flags=p.ER_NO_SEGMENTATION_MASK, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    R_hand_img_arr = p.getCameraImage(width=self.handcamera_width,
                                    height=self.handcamera_height,
                                    viewMatrix=R_handviewMat,
                                    projectionMatrix=R_handprojMat)
    R_hand_rgb = R_hand_img_arr[2]
    R_hand_np_img_arr = np.reshape(R_hand_rgb, (self.hand_camera_height, self.hand_camera_width, 4))

    replaceLines=True
    numRays=11
    rayFrom=[]
    rayTo=[]
    rayHitColor = [1,0,0]
    rayMissColor = [0,1,0]
    rayLen = 0.05
    rayStartLen=0
    numforces=6
    rayTo_list = []
    rayIds_list = []

    #指向角10度の近接覚センサの配列
    for i in range(numforces):
        rayIds=[]
        rayFrom=[]
        rayTo=[]
        for j in range (-int((numRays - 1)/2), int((numRays - 1)/2 + 1)):
            rayFrom.append([rayStartLen, rayStartLen, rayStartLen])
            rayTo.append([0, (rayLen * math.sin(math.radians(j))), -rayLen * math.cos(math.radians(j))])
            #rayTo_R1.append([0, (rayLen * math.sin(math.radians(i))), -rayLen * math.cos(math.radians(i))])
            if (replaceLines):
                rayIds.append(p.addUserDebugLine(rayFrom[j + int((numRays - 1)/2) ], rayTo[j + int((numRays - 1)/2) ],
                rayMissColor,parentObjectUniqueId=self.robonyanUid, parentLinkIndex=self.proximity_list[i]))
                #rayIds_list.append(rayIds)
                #rayIds_R1.append(p.addUserDebugLine(rayFrom[i + int((numRays - 1)/2) ], rayTo_R1[i + int((numRays - 1)/2) ], rayMissColor,parentObjectUniqueId=robonyanUid, parentLinkIndex=proximity_R1))
            else:
                rayIds.append(-1)
                #rayIds_list.append(rayIds)
                #rayIds_R1.append(-1)

        rayIds_list.append(rayIds)

    numThreads=0
    proximity_result_list = []

    for i in range(numforces):
        result=[]
        result = p.rayTestBatch(rayFrom,rayTo,numThreads, parentObjectUniqueId=self.robonyanUid, parentLinkIndex=self.proximity_list[i])
        proximity_result_list.append(result)

        for j in range (numRays):
            hitObjectUid=[]
            hitFraction=[]
            hitPosition=[]
            hitObjectUid = result[j][0]
            hitFraction = result[j][2]
            hitPosition = result[j][3]
            #hitObjectUid_R1=results_R1[i][0]
            #hitFraction_R1 = results_R1[i][2]
            #hitPosition_R1 = results_R1[i][3]
            if (hitFraction==1.):
                p.addUserDebugLine(rayFrom[j],rayTo[j], rayMissColor,replaceItemUniqueId=rayIds_list[i][j],
                parentObjectUniqueId=self.robonyanUid, parentLinkIndex=self.proximity_list[i])
            else:
                localHitTo = [rayFrom[j][0]+hitFraction*(rayTo[j][0]-rayFrom[j][0]),
                                            rayFrom[j][1]+hitFraction*(rayTo[j][1]-rayFrom[j][1]),
                                            rayFrom[i][2]+hitFraction*(rayTo[j][2]-rayFrom[j][2])]
                p.addUserDebugLine(rayFrom[j],localHitTo, rayHitColor,replaceItemUniqueId=rayIds_list[i][j],parentObjectUniqueId=self.robonyanUid, parentLinkIndex=self.proximity_list[i])

    proximity_obsevation = np.array([proximity_list])

    """
    state = p.getLinkState(self.robonyanUid, self.kukaGripperIndex)
    state2 = p.getLinkState(self.robonyanUid, self.kukaGripperIndex2)
    pos = state[0]
    orn = state[1] #クォータニオン
    euler = p.getEulerFromQuaternion(orn) #オイラーに変換

    observation.extend(list(pos))
    observation.extend(list(euler))
    """

    return observation

  def applyAction(self, motorCommands):

    #print ("self.numJoints")
    #print (self.numJoints)
    if (self.useInverseKinematics):

      dx = motorCommands[0]
      dy = motorCommands[1]
      dz = motorCommands[2]
      #da = motorCommands[3]
      droll = motorCommands[3]
      dpitch = motorCommands[4]
      dyaw = motorCommands[5]
      #fingerAngle = motorCommands[4]

      # 両手の手先の状態を取得
      L_state = p.getLinkState(self.robonyanUid, self.L_EndEffectorIndex)
      actualEndEffectorPos = state[0]
      #print("pos[2] (getLinkState(kukaEndEffectorIndex)")
      #print(actualEndEffectorPos[2])

      self.endEffectorPos[0] = self.endEffectorPos[0] + dx
      if (self.endEffectorPos[0] > 0.65):
        self.endEffectorPos[0] = 0.65
      if (self.endEffectorPos[0] < 0.50):
        self.endEffectorPos[0] = 0.50
      self.endEffectorPos[1] = self.endEffectorPos[1] + dy
      if (self.endEffectorPos[1] < -0.17):
        self.endEffectorPos[1] = -0.17
      if (self.endEffectorPos[1] > 0.22):
        self.endEffectorPos[1] = 0.22

      #print ("self.endEffectorPos[2]")
      #print (self.endEffectorPos[2])
      #print("actualEndEffectorPos[2]")
      #print(actualEndEffectorPos[2])
      #if (dz<0 or actualEndEffectorPos[2]<0.5):
      self.endEffectorPos[2] = self.endEffectorPos[2] + dz

      self.endEffectorAngle = self.endEffectorAngle + da
      pos = self.endEffectorPos
      orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
      if (self.useNullSpace == 1):
        if (self.useOrientation == 1):
          jointPoses = p.calculateInverseKinematics(self.robonyanUid, self.kukaEndEffectorIndex, pos,
                                                    orn, self.ll, self.ul, self.jr, self.rp)
        else:
          jointPoses = p.calculateInverseKinematics(self.robonyanUid,
                                                    self.kukaEndEffectorIndex,
                                                    pos,
                                                    lowerLimits=self.ll,
                                                    upperLimits=self.ul,
                                                    jointRanges=self.jr,
                                                    restPoses=self.rp)
      else:
        if (self.useOrientation == 1):
          jointPoses = p.calculateInverseKinematics(self.robonyanUid,
                                                    self.kukaEndEffectorIndex,
                                                    pos,
                                                    orn,
                                                    jointDamping=self.jd)
        else:
          jointPoses = p.calculateInverseKinematics(self.robonyanUid, self.kukaEndEffectorIndex, pos)

      #print("jointPoses")
      #print(jointPoses)
      #print("self.kukaEndEffectorIndex")
      #print(self.kukaEndEffectorIndex)
      if (self.useSimulation):
        for i in range(self.kukaEndEffectorIndex + 1):
          #print(i)
          p.setJointMotorControl2(bodyUniqueId=self.kukaUid,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=jointPoses[i],
                                  targetVelocity=0,
                                  force=self.maxForce,
                                  maxVelocity=self.maxVelocity,
                                  positionGain=0.3,
                                  velocityGain=1)
      else:
        #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        for i in range(self.numJoints):
          p.resetJointState(self.kukaUid, i, jointPoses[i])
      #fingers
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

    else:
      for action in range(len(motorCommands)):
        motor = self.motorIndices[action]
        p.setJointMotorControl2(self.kukaUid,
                                motor,
                                p.POSITION_CONTROL,
                                targetPosition=motorCommands[action],
                                force=self.maxForce)
