import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import pybullet_data
import time
import math
import numpy as np
import random
import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import pybullet_data
from pkg_resources import parse_version



datapath = pybullet_data.getDataPath()

p.connect(p.GUI)

p.resetSimulation()

p.setGravity(0,0,-10)
useRealTimeSim = 0
maxForce = 500.


#p.setTimeStep(1./120.)
#p.setRealTimeSimulation(useRealTimeSim) # either this



p.setAdditionalSearchPath(datapath)
p.loadURDF("plane.urdf", [0, 0, -0.3], useFixedBase=True)

p.loadURDF("table/table.urdf", 1.000000, 0.00000, -0.050000,
           0.000000, 0.000000, 0.707106781187, .707106781187)

#robonyan = p.loadSDF("kuka_iiwa/kuka_with_gripper2.sdf")
robonyan = p.loadSDF("robonyan_gripper/robonyan_test.sdf")
robonyanUid = robonyan[0]
#print(robonyanUid)
p.resetBasePositionAndOrientation(robonyanUid, [-0.100000, 0.000000, 0.30000],
                                  [0.000000, 0.000000, 0.000000, 1.000000])

kinect = p.getLinkState(robonyanUid, 54)
print(kinect)

numJoints = p.getNumJoints(robonyanUid)
print(numJoints)

#p.resetJointState(robonyanUid, 0, 0.785398)
#p.setJointMotorControl2(robonyanUid, 0,p.POSITION_CONTROL, targetPosition = 0.785398, force = maxForce)

#for jointIndex in range(numJoints):
  #p.resetJointState(robonyanUid, jointIndex, 0)
  #p.setJointMotorControl2(robonyanUid,
                          #jointIndex,
                          #p.POSITION_CONTROL,
                          #targetPosition=0,
                          #force=maxForce)

for i in range(numJoints):
  jointInfo = p.getJointInfo(robonyanUid, i)
  print(jointInfo)

proximity_L1 = 10
proximity_L2 = 17
proximity_L3 = 24
proximity_R1 = 37
proximity_R2 = 44
proximity_R3 = 51
proximity_list = np.array([proximity_L1, proximity_L2, proximity_L3, proximity_R1, proximity_R2, proximity_R3])

force_L1 = 12
force_L2 = 19
force_L3 = 26
force_R1 = 39
force_R2 = 46
force_R3 = 53

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

width = 1920
height = 1080

# kinectv2のカメラ座標取得
kinectPos,kinectOrn = kinect[0], kinect[1]
kinectPos = list(kinectPos)
kinectPos[0] += 0.1
kinectPos[1] += 0.002
kinectPos[2] += 0.05
#kinectPos = [-0.112 + 0.03153696, 0, 1.304 + 0.03153696]
kinectPos = tuple(kinectPos)
kinectEuler = p.getEulerFromQuaternion(kinectOrn)
kinectYaw = kinectEuler[2]*360/(2.*math.pi)-90

camInfo = p.getDebugVisualizerCamera()

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
            rayMissColor,parentObjectUniqueId=robonyanUid, parentLinkIndex=proximity_list[i]))
            #rayIds_list.append(rayIds)
            #rayIds_R1.append(p.addUserDebugLine(rayFrom[i + int((numRays - 1)/2) ], rayTo_R1[i + int((numRays - 1)/2) ], rayMissColor,parentObjectUniqueId=robonyanUid, parentLinkIndex=proximity_R1))
        else:
            rayIds.append(-1)
            #rayIds_list.append(rayIds)
            #rayIds_R1.append(-1)

    rayIds_list.append(rayIds)

#for i in range (10000):
while (True):
    p.stepSimulation()
    time.sleep(1./240.)

    numThreads=0
    #results_L1 = p.rayTestBatch(rayFrom,rayTo_L1,numThreads, parentObjectUniqueId=robonyanUid, parentLinkIndex=proximity_L1)
    #results_R1 = p.rayTestBatch(rayFrom,rayTo_R1,numThreads, parentObjectUniqueId=robonyanUid, parentLinkIndex=proximity_R1)
    #result_list = []
    #hitObjectUid_list = []
    #hitFraction_list = []
    #hitPosition_list = []
    result_list = []

    for i in range(numforces):
        result=[]
        result = p.rayTestBatch(rayFrom,rayTo,numThreads, parentObjectUniqueId=robonyanUid, parentLinkIndex=proximity_list[i])
        result_list.append(result)

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
                parentObjectUniqueId=robonyanUid, parentLinkIndex=proximity_list[i])
            else:
                localHitTo = [rayFrom[j][0]+hitFraction*(rayTo[j][0]-rayFrom[j][0]),
                                            rayFrom[j][1]+hitFraction*(rayTo[j][1]-rayFrom[j][1]),
                                            rayFrom[i][2]+hitFraction*(rayTo[j][2]-rayFrom[j][2])]
                p.addUserDebugLine(rayFrom[j],localHitTo, rayHitColor,replaceItemUniqueId=rayIds_list[i][j],parentObjectUniqueId=robonyanUid, parentLinkIndex=proximity_list[i])

            #if (hitFraction_R1==1.):
                #p.addUserDebugLine(rayFrom[i],rayTo_R1[i], rayMissColor,replaceItemUniqueId=rayIds_R1[i],parentObjectUniqueId=robonyanUid, parentLinkIndex=proximity_R1)
            #else:
                #localHitTo_R1 = [rayFrom[i][0]+hitFraction_R1*(rayTo_R1[i][0]-rayFrom[i][0]),
                                        #rayFrom[i][1]+hitFraction_R1*(rayTo_R1[i][1]-rayFrom[i][1]),
                                        #rayFrom[i][2]+hitFraction_R1*(rayTo_R1[i][2]-rayFrom[i][2])]
                #p.addUserDebugLine(rayFrom[i],localHitTo_R1, rayHitColor,replaceItemUniqueId=rayIds_R1[i],parentObjectUniqueId=robonyanUid, parentLinkIndex=proximity_R1)

    #print(np.array([results_L1]))
    #print(np.array([results_R1]))
    #print(np.array(result_list))
    #for k in range(6):
        #print("%d:" % k, np.array(result_list)[k])

    kinectMat = p.getMatrixFromQuaternion(kinectOrn)
    upVector = [0,0,1]
    forwardVec = [kinectMat[0],kinectMat[3],kinectMat[6]]
    #sideVec =  [camMat[1],camMat[4],camMat[7]]
    kinectUpVec =  [kinectMat[2],kinectMat[5],kinectMat[8]]
    kinectTarget = [kinectPos[0]+forwardVec[0]*50,kinectPos[1]+forwardVec[1]*50,kinectPos[2]+forwardVec[2]*50]
    kinectUpTarget = [kinectPos[0]+kinectUpVec[0],kinectPos[1]+kinectUpVec[1],kinectPos[2]+kinectUpVec[2]]
    kinectviewMat = p.computeViewMatrix(kinectPos, kinectTarget, kinectUpVec)
    kinectprojMat = camInfo[3]
    #p.getCameraImage(320,200,viewMatrix=viewMat,projectionMatrix=projMat, flags=p.ER_NO_SEGMENTATION_MASK, renderer=p.ER_BULLET_HARDWARE_OPENGL)
    p.getCameraImage(width,height,viewMatrix=kinectviewMat,projectionMatrix=kinectprojMat, renderer=p.ER_BULLET_HARDWARE_OPENGL)



    ContactPoints_L1 = p.getContactPoints(robonyanUid, -1, force_L1, -1)
    ContactPoints_L2 = p.getContactPoints(robonyanUid, -1, force_L2, -1)
    ContactPoints_L3 = p.getContactPoints(robonyanUid, -1, force_L3, -1)
    ContactPoints_R1 = p.getContactPoints(robonyanUid, -1, force_R1, -1)
    ContactPoints_R2 = p.getContactPoints(robonyanUid, -1, force_R2, -1)
    ContactPoints_R3 = p.getContactPoints(robonyanUid, -1, force_R3, -1)
    #print(np.array([ContactPoints_L3]))
    #print(np.array([ContactPoints_R3]))


p.disconnect()
