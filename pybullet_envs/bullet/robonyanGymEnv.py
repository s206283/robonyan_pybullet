import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
from . import robonyan
import random
import pybullet_data
from pkg_resources import parse_version

maxSteps = 1000

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class RobonyanGymEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False):
    self._timeStep = 1. / 240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    self._envStepCounter = 0
    self._renders = renders
    self._width = 341
    self._height = 256
    self._isDiscrete = isDiscrete
    self.terminated = 0
    self._p = p
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid < 0):
        p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3, 180, -41, [0.52, -0.2, -0.33])
    else:
      p.connect(p.DIRECT)
    #timinglog = p.startStateLogging(p.STATE_LOGGING_PROFILE_TIMINGS, "kukaTimings.json")
    self.seed()
    self.reset()
    observationDim = len(self.getExtendedObservation())
    #print("observationDim")
    #print(observationDim)

    observation_high = np.array([np.finfo(np.float32).max] * observationDim)
    if (self._isDiscrete):
      self.action_space = spaces.Discrete(7) # Discreteは離散値、連続値はBoxクラスを使う
    else:
      action_dim = 3
      self._action_bound = 1
      action_high = np.array([self._action_bound] * action_dim)
      self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
    self.observation_space = spaces.Box(low=0,
                                        high=255,
                                        shape=(self._height, self._width, 4),
                                        dtype=np.uint8)
    self.viewer = None

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

  def reset(self):
    self.terminated = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -1])

    p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 0.5000000, 0.00000, -.820000,
               0.000000, 0.000000, 0.0, 1.0)

    xpos = 0.5 + 0.2 * random.random()
    ypos = 0 + 0.25 * random.random()
    ang = 3.1415925438 * random.random()
    orn = p.getQuaternionFromEuler([0, 0, ang])
    self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, -0.1,
                               orn[0], orn[1], orn[2], orn[3])

    p.setGravity(0, 0, -10)
    self._robonyan = robonyan.robonyan(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    self._envStepCounter = 0
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation)

  def __del__(self):
    p.disconnect()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):

    #camEyePos = [0.03,0.236,0.54]
    #distance = 1.06
    #pitch=-56
    #yaw = 258
    #roll=0
    #upAxisIndex = 2
    #camInfo = p.getDebugVisualizerCamera()
    #print("width,height")
    #print(camInfo[0])
    #print(camInfo[1])
    #print("viewMatrix")
    #print(camInfo[2])
    #print("projectionMatrix")
    #print(camInfo[3])
    #viewMat = camInfo[2]
    #viewMat = p.computeViewMatrixFromYawPitchRoll(camEyePos,distance,yaw, pitch,roll,upAxisIndex)
    """
    viewMat = [
        -0.5120397806167603, 0.7171027660369873, -0.47284144163131714, 0.0, -0.8589617609977722,
        -0.42747554183006287, 0.28186774253845215, 0.0, 0.0, 0.5504802465438843,
        0.8348482847213745, 0.0, 0.1925382763147354, -0.24935829639434814, -0.4401884973049164, 1.0
    ]
    #projMatrix = camInfo[3]#[0.7499999403953552, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0, -0.02000020071864128, 0.0]
    projMatrix = [
        0.75, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, -1.0000200271606445, -1.0, 0.0, 0.0,
        -0.02000020071864128, 0.0
    ]

    img_arr = p.getCameraImage(width=self._width,
                               height=self._height,
                               viewMatrix=viewMat,
                               projectionMatrix=projMatrix)
    rgb = img_arr[2]
    np_img_arr = np.reshape(rgb, (self._height, self._width, 4))
    self._observation = np_img_arr
    """

    self._observation = []
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

    self._kinect_observation = kinect_np_img_arr

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

    self._L_observation = L_hand_np_img_arr

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

    self._R_observation = R_hand_np_img_arr

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
        proximity_result_list.append(result[2])

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

    self._proximity_obsevation = np.array([proximity_result_list])

    """
    state = p.getLinkState(self.robonyanUid, self.kukaGripperIndex)
    state2 = p.getLinkState(self.robonyanUid, self.kukaGripperIndex2)
    pos = state[0]
    orn = state[1] #クォータニオン
    euler = p.getEulerFromQuaternion(orn) #オイラーに変換

    observation.extend(list(pos))
    observation.extend(list(euler))
    """


    return self._kinect_observation, self._L_observation, self._R_observation, self._proximity_obsevation

  def step(self, action):
    if (self._isDiscrete):
      dv = 0.01
      dx = [0, -dv, dv, 0, 0, 0, 0][action]
      dy = [0, 0, 0, -dv, dv, 0, 0][action]
      da = [0, 0, 0, 0, 0, -0.1, 0.1][action]
      f = 0.3
      realAction = [dx, dy, -0.002, da, f]
    else:
      dv = 0.01
      dx = action[0] * dv
      dy = action[1] * dv
      dz = action[2] * dv
      #da = action[3] * 0.1
      droll = action[3]
      dpitch = action[4]
      dyaw = action[5]
      #f = 0.3
      realAction = [dx, dy, dz, droll, dpitch, dyaw]

    return self.step2(realAction)

  def step2(self, action):
    for i in range(self._actionRepeat):
      self._robonyan.applyAction(action)
      p.stepSimulation()
      if self._termination():
        break
      #self._observation = self.getExtendedObservation()
      self._envStepCounter += 1

    self._observation = self.getExtendedObservation()
    if self._renders:
      time.sleep(self._timeStep)

    #print("self._envStepCounter")
    #print(self._envStepCounter)

    done = self._termination()
    reward = self._reward()
    #print("len=%r" % len(self._observation))

    return np.array(self._observation), reward, done, {}

  def render(self, mode='human', close=False):
    if mode != "rgb_array":
      return np.array([])
    base_pos, orn = self._p.getBasePositionAndOrientation(self._racecar.racecarUniqueId)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=base_pos,
                                                            distance=self._cam_dist,
                                                            yaw=self._cam_yaw,
                                                            pitch=self._cam_pitch,
                                                            roll=0,
                                                            upAxisIndex=2)
    proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                     aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                     nearVal=0.1,
                                                     farVal=100.0)
    (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                              height=RENDER_HEIGHT,
                                              viewMatrix=view_matrix,
                                              projectionMatrix=proj_matrix,
                                              renderer=pybullet.ER_BULLET_HARDWARE_OPENGL)
    rgb_array = np.array(px)
    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _termination(self): # リーチング後の動作
    #print (self._kuka.endEffectorPos[2])
    state = p.getLinkState(self._robonyan.robonyanUid, self._robonyan.robonyanEndEffectorIndex)
    actualEndEffectorPos = state[0]

    #print("self._envStepCounter")
    #print(self._envStepCounter)
    if (self.terminated or self._envStepCounter > maxSteps):
      self._observation = self.getExtendedObservation()
      return True
    maxDist = 0.005
    closestPoints = p.getClosestPoints(self._kuka.trayUid, self._kuka.kukaUid, maxDist)

    if (len(closestPoints)):  #(actualEndEffectorPos[2] <= -0.43):
      self.terminated = 1

      #print("closing gripper, attempting grasp")
      #start grasp and terminate
      fingerAngle = 0.3
      for i in range(100):
        graspAction = [0, 0, 0.0001, 0, fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        fingerAngle = fingerAngle - (0.3 / 100.)
        if (fingerAngle < 0):
          fingerAngle = 0

      for i in range(1000):
        graspAction = [0, 0, 0.001, 0, fingerAngle]
        self._kuka.applyAction(graspAction)
        p.stepSimulation()
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        if (blockPos[2] > 0.23):
          #print("BLOCKPOS!")
          #print(blockPos[2])
          break
        state = p.getLinkState(self._kuka.kukaUid, self._kuka.kukaEndEffectorIndex)
        actualEndEffectorPos = state[0]
        if (actualEndEffectorPos[2] > 0.5):
          break

      self._observation = self.getExtendedObservation()
      return True
    return False

  def _reward(self):

    #rewards is height of target object
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    closestPoints = p.getClosestPoints(self.blockUid, self._kuka.kukaUid, 1000, -1,
                                       self._kuka.kukaEndEffectorIndex)

    reward = -1000
    numPt = len(closestPoints)
    #print(numPt)
    if (numPt > 0):
      #print("reward:")
      reward = -closestPoints[0][8] * 10
    if (blockPos[2] > 0.2):
      #print("grasped a block!!!")
      #print("self._envStepCounter")
      #print(self._envStepCounter)
      reward = reward + 1000

    #print("reward")
    #print(reward)
    return reward

  if parse_version(gym.__version__) < parse_version('0.9.6'):
    _render = render
    _reset = reset
    _seed = seed
    _step = step
