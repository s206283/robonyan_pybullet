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
import pdb
import distutils.dir_util
import glob
import pybullet_data
from pkg_resources import parse_version

maxSteps = 1000

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

class RobonyanGraspGymEnv(gym.Env):
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
        #self._cam_dist = 1.3
        #self._cam_yaw = 180
        #self._cam_pitch = -40
        self._cam_dist = 0.2
        #self._cam_yaw = 45
        self._cam_roll = -45
        self._cam_yaw = 180
        #self._cam_pitch = -45
        self._cam_pitch = 0
        #self._width = 341
        #self._height = 256
        self._kinect_rgb_width = 1920
        self._kinect_rgb_height = 1080
        self._kinect_d_width = 512
        self._kinect_d_height = 424

        self._handcamera_width = 640
        self._handcamera_height = 480

        self._isDiscrete = isDiscrete
        #self._isBox = isBox
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
            self.action_space = spaces.Discrete(7)
        else:
            action_dim = 12
            self._action_bound = 1
            action_high = np.array([self._action_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
            self._proximity_low = np.array([0] * 3)
            self._proximity_high = np.array([1] * 3)
            self._force_low = np.array([0] * 3)
            self._force_high = np.array([10] * 3)

        self.observation_space = spaces.Tuple((spaces.Box(low=0,high=255,
                                                         shape=(self._kinect_rgb_height, self._kinect_rgb_width, 4),
                                                         dtype=np.uint8),
                                              spaces.Box(self._proximity_low, self._proximity_high, dtype=np.float32),
                                              spaces.Box(self._force_low, self._force_high, dtype=np.float32)))

        self.viewer = None

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

    def reset(self):
        self.terminated = 0
        self._grasp = False
        self._close = False
        self._env_step = 0
        #physicsClient = p.connect(p.GUI)
        p.setAdditionalSearchPath(self._urdfRoot)
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timeStep)
        #p.loadURDF(os.path.join(self._urdfRoot, "plane.urdf"), [0, 0, -0.3])
        p.loadURDF("plane.urdf", [0, 0, -0.3], useFixedBase=True)

        p.loadURDF(os.path.join(self._urdfRoot, "table/table.urdf"), 1.000000, 0.00000, -0.050000,
                   0.000000, 0.000000, 0.707106781187, .707106781187)

        xpos = 0.8 + 0.1 * random.random()
        ypos = 0 - 0.6 * random.random()
        ang = 3.1415925438 * random.random()
        orn = p.getQuaternionFromEuler([0, 0, ang])

        """
        self.blockUid = p.loadURDF(os.path.join(self._urdfRoot, "block.urdf"), xpos, ypos, 0.58,
                                   orn[0], orn[1], orn[2], orn[3])
        """

        p.setGravity(0, 0, -10)
        self._robonyan = robonyan.Robonyan(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)

        self._envStepCounter = 0
        p.stepSimulation()

        urdf_pattern = os.path.join(self._urdfRoot, 'random_urdfs/00[0-9]/*.urdf')
        found_object_directories = glob.glob(urdf_pattern)
        total_num_objects = len(found_object_directories)
        selected_object = np.random.choice(np.arange(total_num_objects))
        selected_object_filename = found_object_directories[selected_object]

        self.blockUid = p.loadURDF((selected_object_filename),xpos, ypos, 0.58,
                                    orn[0], orn[1], orn[2], orn[3])

        for _ in range(500):
          p.stepSimulation()

        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        self.blockPos, self.blockOrn = blockPos, blockOrn
        #blockPos = np.array(blockPos)

        # noiseを加える
        R_position = [blockPos[0], blockPos[1], blockPos[2] + 0.1, 0, 0.785398, 0]
        L_position = [blockPos[0], blockPos[1] + 0.4, blockPos[2] + 0.1, 0, 0, -1.5708]

        self._robonyan.resetHandPosition(R_position, L_position)

        p.stepSimulation()

        self._observation = self.getExtendedObservation()
        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):

        camInfo = p.getDebugVisualizerCamera()

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.blockPos,
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
                                                                roll=self._cam_roll,
                                                                upAxisIndex=2)

        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                  height=RENDER_HEIGHT,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = np.reshape(px, (RENDER_HEIGHT, RENDER_WIDTH, 4))
        rgb_array = rgb_array[:, :, :3]

        self.image = rgb_array

        kinectviewMat = self.kinectviewMat()
        kinectprojMat = camInfo[3]
        #p.getCameraImage(320,200,viewMatrix=viewMat,projectionMatrix=projMat, flags=p.ER_NO_SEGMENTATION_MASK, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        #p.getCameraImage(width,height,viewMatrix=kinectviewMat,projectionMatrix=kinectprojMat, renderer=p.ER_BULLET_HARDWARE_OPENGL)

        kinect_img_arr = p.getCameraImage(width=self._kinect_rgb_width,
                                          height=self._kinect_rgb_height,
                                          viewMatrix=kinectviewMat,
                                          projectionMatrix=kinectprojMat)
        kinect_rgb = kinect_img_arr[2]
        # HxWxC
        kinect_np_img_arr = np.reshape(kinect_rgb, (self._kinect_rgb_height, self._kinect_rgb_width, 4))
        # C : 4 → 3
        kinect_np_img_arr = kinect_np_img_arr[:, :, :3]
        # CxHxW
        #kinect_np_img_arr = kinect_np_img_arr.transpose((2, 0, 1))
        #kinect_flatten = kinect_np_img_arr.reshape(-1)

        self._kinect_observation = kinect_np_img_arr
        #print(self._kinect_observation.shape)

        L_handviewMat = self.L_handviewMat()
        L_handprojMat = camInfo[3]
        #p.getCameraImage(320,200,viewMatrix=viewMat,projectionMatrix=projMat, flags=p.ER_NO_SEGMENTATION_MASK, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        L_hand_img_arr = p.getCameraImage(width=self._handcamera_width,
                                          height=self._handcamera_height,
                                          viewMatrix=L_handviewMat,
                                          projectionMatrix=L_handprojMat)
        L_hand_rgb = L_hand_img_arr[2]
        # HxWxC
        L_hand_np_img_arr = np.reshape(L_hand_rgb, (self._handcamera_height, self._handcamera_width, 4))
        # C : 4 → 3
        L_hand_np_img_arr = L_hand_np_img_arr[:, :, :3]
        # CxHxW
        #L_hand_np_img_arr = L_hand_np_img_arr.transpose((2, 0, 1))

        self._L_observation = L_hand_np_img_arr

        #R_hand視点
        R_handviewMat = self.R_handviewMat()
        R_handprojMat = camInfo[3]
        #p.getCameraImage(320,200,viewMatrix=viewMat,projectionMatrix=projMat, flags=p.ER_NO_SEGMENTATION_MASK, renderer=p.ER_BULLET_HARDWARE_OPENGL)
        R_hand_img_arr = p.getCameraImage(width=self._handcamera_width,
                                          height=self._handcamera_height,
                                          viewMatrix=R_handviewMat,
                                          projectionMatrix=R_handprojMat)
        R_hand_rgb = R_hand_img_arr[2]
        # HxWxC
        R_hand_np_img_arr = np.reshape(R_hand_rgb, (self._handcamera_height, self._handcamera_width, 4))
        # C : 4 → 3
        R_hand_np_img_arr = R_hand_np_img_arr[:, :, :3]
        # CxHxW
        #R_hand_np_img_arr = R_hand_np_img_arr.transpose((2, 0, 1))

        self._R_observation = R_hand_np_img_arr

        self._L_proximity_observation, self._R_proximity_observation = self.proximity_sensor()

        self._L_contact_observation, self._R_contact_observation = self.force_sensor()

        self.L_proprioception, self.R_proprioception = self.proprioception()

        #print(self._L_proximity_observation)
        #print(self._L_proximity_observation.shape)
        #print(self._R_proximity_observation)
        #print(self._R_proximity_observation.shape)

        #self._observation = (self._R_proximity_observation, self._R_contact_observation)
        self._observation = np.concatenate([self._R_proximity_observation,
                                            self._R_contact_observation,
                                            self.R_proprioception], 0)
        #print(self._observation)

        self._observation = (self.image, self._observation)

        return self._observation

    def kinectviewMat(self):
        # kinectv2のカメラ座標取得
        kinect = p.getLinkState(self._robonyan.robonyanUid, 56)
        kinectPos,kinectOrn = kinect[0], kinect[1]
        kinectPos = list(kinectPos)
        kinectPos[0] += 0.2
        kinectPos[1] += 0.002
        kinectPos[2] += 0.05
        #kinectPos = [-0.112 + 0.03153696, 0, 1.304 + 0.03153696]
        kinectPos = tuple(kinectPos)

        camInfo = p.getDebugVisualizerCamera()

        kinectMat = p.getMatrixFromQuaternion(kinectOrn)
        upVector = [0,0,1]
        forwardVec = [kinectMat[0],kinectMat[3],kinectMat[6]]
        #sideVec =  [camMat[1],camMat[4],camMat[7]]
        kinectUpVec =  [kinectMat[2],kinectMat[5],kinectMat[8]]
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        kinectTarget = blockPos
        kinectUpTarget = [kinectPos[0]+kinectUpVec[0],kinectPos[1]+kinectUpVec[1],kinectPos[2]+kinectUpVec[2]]
        kinectviewMat = p.computeViewMatrix(kinectPos, kinectTarget, kinectUpVec)

        return kinectviewMat


    def L_handviewMat(self):
        L_hand = p.getLinkState(self._robonyan.robonyanUid, 6)

        # L_handのカメラ座標取得
        L_handPos,L_handOrn = L_hand[0], L_hand[1]
        L_handEuler = p.getEulerFromQuaternion(L_handOrn)
        L_handYaw = L_handEuler[2]*360/(2.*math.pi)-90

        L_camera = np.array([0.02, 0, 0])
        L_handOrn = list(L_handOrn)
        # 回転行列の生成
        Lx = self.rotate_x(L_handOrn[0])
        Ly = self.rotate_y(L_handOrn[1])
        Lz = self.rotate_z(L_handOrn[2])
        # 回転後のベクトルを計算
        L1 = np.dot(Ly,Lx)
        L2 = np.dot(L1,Lz)
        L3 = np.dot(L2,L_camera)
        # 途中経過
        L4 = np.dot(Lz,L_camera)
        La = np.dot(Lx,Lz)
        L5 = np.dot(La,L_camera)
        # 整数型に変換
        b = np.array(L3,dtype=np.float32)
        L_handPos = np.array(L_handPos) + b
        L_handPos = L_handPos.tolist()
        L_handPos = L_handPos[0]
        #print(R_handPos)
        L_handPos = tuple(L_handPos)
        #print(R_handPos)
        L4 = np.array(L4,dtype=np.float32)
        L5 = np.array(L5,dtype=np.float32)

        L_handOrn = tuple(L_handOrn)

        camInfo = p.getDebugVisualizerCamera()

        L_handMat = p.getMatrixFromQuaternion(L_handOrn)
        upVector = [0,0,1]
        forwardVec = [L_handMat[0],L_handMat[3],L_handMat[6]]
        #sideVec =  [camMat[1],camMat[4],camMat[7]]
        L_handUpVec =  [L_handMat[2],L_handMat[5],L_handMat[8]]
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        L_handTarget = self.blockPos
        L_handUpTarget = [L_handPos[0]+L_handUpVec[0],L_handPos[1]+L_handUpVec[1],L_handPos[2]+L_handUpVec[2]]
        L_handviewMat = p.computeViewMatrix(L_handPos, L_handTarget, L_handUpVec)

        return L_handviewMat

    def R_handviewMat(self):

        R_hand = p.getLinkState(self._robonyan.robonyanUid, 34)

        # R_handのカメラ座標取得
        R_handPos,R_handOrn = R_hand[0], R_hand[1]
        R_handEuler = p.getEulerFromQuaternion(R_handOrn)
        R_handYaw = R_handEuler[2]*360/(2.*math.pi)-90
        R_camera = np.array([0.02, 0, 0])
        R_handOrn = list(R_handOrn)
        # 回転行列の生成
        Rx = self.rotate_x(R_handOrn[0])
        Ry = self.rotate_y(R_handOrn[1])
        Rz = self.rotate_z(R_handOrn[2])
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

        #R_hand視点
        R_handMat = p.getMatrixFromQuaternion(R_handOrn)
        upVector = [0,0,1]
        forwardVec = [R_handMat[0],R_handMat[3],R_handMat[6]]
        #sideVec =  [camMat[1],camMat[4],camMat[7]]
        R_handUpVec =  [R_handMat[2],R_handMat[5],R_handMat[8]]
        #R_handTarget = [R_handPos[0]+forwardVec[0]*100,R_handPos[1]+forwardVec[1]*100,R_handPos[2]+forwardVec[2]*100]
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        R_handTarget = blockPos
        R_handUpTarget = [R_handPos[0]+R_handUpVec[0],R_handPos[1]+R_handUpVec[1],R_handPos[2]+R_handUpVec[2]]
        R_handviewMat = p.computeViewMatrix(R_handPos, R_handTarget, R_handUpVec)

        return R_handviewMat

    def proximity_sensor(self):

        replaceLines=True
        numRays=11
        rayFrom=[]
        rayTo=[]
        rayHitColor = [1,0,0]
        rayMissColor = [0,1,0]
        rayLen = 0.5
        rayStartLen=0
        numforces=6
        rayTo_list = []
        rayIds_list = []
        proximity_min = 4
        proximity_max = 50

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
                                                     rayMissColor,parentObjectUniqueId=self._robonyan.robonyanUid,
                                                     parentLinkIndex=self._robonyan.proximity_list[i]))
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
            result = p.rayTestBatch(rayFrom,rayTo,numThreads,
                                    parentObjectUniqueId=self._robonyan.robonyanUid,
                                    parentLinkIndex=self._robonyan.proximity_list[i])
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
                                       parentObjectUniqueId=self._robonyan.robonyanUid,
                                       parentLinkIndex=self._robonyan.proximity_list[i])
                else:
                    localHitTo = [rayFrom[j][0]+hitFraction*(rayTo[j][0]-rayFrom[j][0]),
                                  rayFrom[j][1]+hitFraction*(rayTo[j][1]-rayFrom[j][1]),
                                  rayFrom[i][2]+hitFraction*(rayTo[j][2]-rayFrom[j][2])]
                    p.addUserDebugLine(rayFrom[j],localHitTo, rayHitColor,
                                       replaceItemUniqueId=rayIds_list[i][j],
                                       parentObjectUniqueId=self._robonyan.robonyanUid,
                                       parentLinkIndex=self._robonyan.proximity_list[i])

        #proximity_obsevation = np.array([proximity_result_list])
        #proximity_result = np.array(proximity_result_list)
        #print(proximity_result)
        #print(proximity_result.shape)
        L_proximity_list = proximity_result_list[:3]
        R_proximity_list = proximity_result_list[3:]
        #L_proximity_observation = np.array(L_proximity_list)
        #R_proximity_observation = np.array(R_proximity_list)

        L_proximity = []
        R_proximity = []

        for i in range(3):
            L_proximity_finger = []
            R_proximity_finger = []
            for j in range(11):
                L_proximity_clip = np.clip(L_proximity_list[i][j][2] * proximity_max,
                                            proximity_min, proximity_max)
                R_proximity_clip = np.clip(R_proximity_list[i][j][2] * proximity_max,
                                            proximity_min, proximity_max)
                L_proximity_norm = (L_proximity_clip-proximity_min)/(proximity_max-proximity_min)
                R_proximity_norm = (R_proximity_clip-proximity_min)/(proximity_max-proximity_min)
                #L_proximity_finger.append(L_proximity_list[i][j][2])
                #R_proximity_finger.append(R_proximity_list[i][j][2])
                L_proximity_finger.append(L_proximity_norm)
                R_proximity_finger.append(R_proximity_norm)
            # 11のrayのmin
            #L_proximity.append(sum(L_proximity_finger)/len(L_proximity_finger))
            #R_proximity.append(sum(R_proximity_finger)/len(R_proximity_finger))
            L_proximity.append(min(L_proximity_finger))
            R_proximity.append(min(R_proximity_finger))

        L_proximity_observation = np.array(L_proximity)
        R_proximity_observation = np.array(R_proximity)

        return L_proximity_observation, R_proximity_observation

    def force_sensor(self):
        ContactPoints_L1 = p.getContactPoints(self._robonyan.robonyanUid, -1,
                                            self._robonyan.force_L1, -1)
        ContactPoints_L2 = p.getContactPoints(self._robonyan.robonyanUid, -1,
                                            self._robonyan.force_L2, -1)
        ContactPoints_L3 = p.getContactPoints(self._robonyan.robonyanUid, -1,
                                            self._robonyan.force_L3, -1)
        ContactPoints_R1 = p.getContactPoints(self._robonyan.robonyanUid, -1,
                                            self._robonyan.force_R1, -1)
        ContactPoints_R2 = p.getContactPoints(self._robonyan.robonyanUid, -1,
                                            self._robonyan.force_R2, -1)
        ContactPoints_R3 = p.getContactPoints(self._robonyan.robonyanUid, -1,
                                            self._robonyan.force_R3, -1)


        if len(ContactPoints_L1) > 0:
            ContactPoints_L1 = ContactPoints_L1[0][9]
            ContactPoints_L1 = np.random.normal(ContactPoints_L1, ContactPoints_L1 * 0.02)
        else:
            ContactPoints_L1 = 0
        if len(ContactPoints_L2) > 0:
            ContactPoints_L2 = ContactPoints_L2[0][9]
            ContactPoints_L2 = np.random.normal(ContactPoints_L2, ContactPoints_L2 * 0.02)
        else:
            ContactPoints_L2 = 0
        if len(ContactPoints_L3) > 0:
            ContactPoints_L3 = ContactPoints_L3[0][9]
            ContactPoints_L3 = np.random.normal(ContactPoints_L3, ContactPoints_L3 * 0.02)
        else:
            ContactPoints_L3 = 0
        if len(ContactPoints_R1) > 0:
            ContactPoints_R1 = ContactPoints_R1[0][9]
            ContactPoints_R1 = np.random.normal(ContactPoints_R1, ContactPoints_R1 * 0.02)
        else:
            ContactPoints_R1 = 0
        if len(ContactPoints_R2) > 0:
            ContactPoints_R2 = ContactPoints_R2[0][9]
            ContactPoints_R2 = np.random.normal(ContactPoints_R2, ContactPoints_R2 * 0.02)
        else:
            ContactPoints_R2 = 0
        if len(ContactPoints_R3) > 0:
            ContactPoints_R3 = ContactPoints_R3[0][9]
            ContactPoints_R3 = np.random.normal(ContactPoints_R3, ContactPoints_R3 * 0.02)
        else:
            ContactPoints_R3 = 0


        L_contactlist = [ContactPoints_L1, ContactPoints_L2, ContactPoints_L3]
        R_contactlist = [ContactPoints_R1, ContactPoints_R2, ContactPoints_R3]

        L_contact_observation = np.array(L_contactlist)
        R_contact_observation = np.array(R_contactlist)

        return L_contact_observation, R_contact_observation

    def proprioception(self):
        L_proprioception = []
        R_proprioception = []
        for i in range(12):
            motor = self._robonyan.Lfinger_list[i]
            joint_state = p.getJointState(self._robonyan.robonyanUid,motor)[0]
            L_proprioception.append(joint_state)

        for i in range(12):
            motor = self._robonyan.Rfinger_list[i]
            joint_state = p.getJointState(self._robonyan.robonyanUid,motor)[0]
            R_proprioception.append(joint_state)

        L_proprioception = np.array(L_proprioception)
        R_proprioception = np.array(R_proprioception)

        return L_proprioception, R_proprioception




    def step(self, action):
        assert action.shape == (12,)
        for i in range(self._actionRepeat): # self._actionRepeat = 1
            self._robonyan.graspAction(action)
            p.stepSimulation()
            if self._termination():
                break
            #self._observation = self.getExtendedObservation()
            self._envStepCounter += 1

            self._observation = self.getExtendedObservation()
            if (1 - abs((self._observation[1][3] + self._observation[1][4]) - self._observation[1][5])/5) =1:
                self._grasp = True

            if self._renders:
                time.sleep(self._timeStep)

            #print("self._envStepCounter")
            #print(self._envStepCounter)

            reward = self._reward()
            done = self._termination()

            #print("len=%r" % len(self._observation))

            return np.array(self._observation), reward, done, {}

    def render(self, mode='human', close=False):
        if mode != "rgb_array":
            return np.array([])
        #base_pos, orn = self._p.getBasePositionAndOrientation(self.blockUid)

        #block = p.getLinkState(self.blockUid, 0)
        #blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        #blockPos,blockOrn = block[0], block[1]

        view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=self.blockPos,
                                                                distance=self._cam_dist,
                                                                yaw=self._cam_yaw,
                                                                pitch=self._cam_pitch,
                                                                roll=self._cam_roll,
                                                                upAxisIndex=2)
        #view_matrix = self.kinectviewMat()
        #view_matrix = self.L_handviewMat()
        proj_matrix = self._p.computeProjectionMatrixFOV(fov=60,
                                                         aspect=float(RENDER_WIDTH) / RENDER_HEIGHT,
                                                         nearVal=0.1,
                                                         farVal=100.0)
        (_, _, px, _, _) = self._p.getCameraImage(width=RENDER_WIDTH,
                                                  height=RENDER_HEIGHT,
                                                  viewMatrix=view_matrix,
                                                  projectionMatrix=proj_matrix,
                                                  renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        rgb_array = np.array(px)
        rgb_array = rgb_array[:, :, :3]
        return rgb_array

    def _termination(self):
        """Terminates the episode if we have tried to grasp or if we are above
        maxSteps steps.
        """
        return self._grasp or self._close or self._envStepCounter >= maxSteps

    def _reward(self):

        reward = 0

        if (self._observation[1][3] + self._observation[1][4] + self._observation[1][5]) > 20:
            reward = -1
            self._close = True
            return reward

        if abs((self._observation[1][3] + self._observation[1][4]) - self._observation[1][5]) < 0.1 :
            reward = 1 - (abs(self._observation[1][5] - 5) / 5)**0.4

        #print("reward")
        #print(reward)

        if reward > 0.8:
            self._grasp = True
        return reward

    if parse_version(gym.__version__) < parse_version('0.9.6'):
        _render = render
        _reset = reset
        _seed = seed
        _step = step
