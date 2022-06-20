import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import time
import pybullet_data
import numpy as np
import copy
import math
import random
import braccio_arm_inverse_kinematics as inverse

class braccio_arm_v0:

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
        self.baEndEffectorIndex = 5
        self.baGripperIndex = 5
        self.baFingerIndexL = 6
        self.baFingerIndexR = 7 #check urdf file with the index number
        # lower limits for null space
        self.ll = [-.967, -2, -2.96, 0.19, -2.96, -2.09, -3.05]
       

        # upper limits for null space
        self.ul = [.967, 2, 2.96, 2.29, 2.96, 2.09, 3.05]
    
        # joint ranges for null space
        self.jr = [5.8, 4, 5.8, 4, 5.8, 4, 6]
        
        # restposes for null space
        #self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        self.rp =  [1.5708, 1.5708, 1.5708, 1.5708, 
        1.5708, 1.27409, 1.27409]
        # joint damping coefficents
        self.jd = None
        self.reset()

    def reset(self):
        #load model
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        robot = p.loadURDF("/home/jessie/internship/model/braccio_arm_clean.urdf") 
        # get barobot id
        self.baUid = robot
        # reset original position
        p.resetBasePositionAndOrientation(self.baUid, [0.0, 0.0, 0.0], # position of robot, GREEN IS Y AXIS
                                      [0.000000, 0.000000, 0.000000, 1.000000]) # direction of robot
        self.jointPositions = [0.0, 1.5708, 1.5708, 1.5708, 1.5708, 
        1.5708, 1.27409, 1.27409] # 8 joints(including world joint) random reset joint angles
        # base     (M1): 90 degrees Allowed values from 0 to 180 degrees
        # Shoulder (M2): 90 degrees Allowed values from 15 to 165 degrees
        # Elbow    (M3): 90 degrees Allowed values from 0 to 180 degrees
        # Wrist    (M4): 90 degrees Allowed values from 0 to 180 degrees
        # Wrist rot(M5): 90 degrees Allowed values from 0 to 180 degrees
        # gripper  (M6): 73 degrees Allowed values from 10 to 73 degrees. 10: the toungue is open, 73: the gripper is closed.


        self.numJoints = p.getNumJoints(self.baUid)
        for jointIndex in range(self.numJoints):
            p.resetJointState(self.baUid, jointIndex, self.jointPositions[jointIndex])
            p.setJointMotorControl2(self.baUid,
                              jointIndex,
                              p.POSITION_CONTROL,
                              targetPosition=self.jointPositions[jointIndex],
                              force=self.maxForce)

        self.endEffectorPos = [0, 0, 0] #TODO
    
        self.endEffectorAngle = 0 #0.02 #TODO

        self.motorNames = []
        self.motorIndices = []

        for i in range(self.numJoints):
            jointInfo = p.getJointInfo(self.baUid, i)
            qIndex = jointInfo[3]
            if qIndex > -1:
                self.motorNames.append(str(jointInfo[1]))
                self.motorIndices.append(i)

    def getActionDimension(self):
        if (self.useInverseKinematics):
            return len(self.motorIndices)
        return 6  #position x,y,z and roll/pitch/yaw euler angles of end effector

    def getObservationDimension(self):
        return len(self.getObservation())

    def getObservation(self):
        observation = []
        # state for gripper(end effector)
        state = p.getLinkState(self.baUid, self.baGripperIndex)
        pos = state[4]
        pos=list(pos)
        orn = state[5]

        euler = p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(euler))

        return observation

    def get_to_place(self,position):
        orn=[]
        jointPoses = inverse.getinversePoisition(self.baUid,position,orn)
        p.resetBasePositionAndOrientation(self.baUid,[0.0, 0.0, 0.0], # position of robot, GREEN IS Y AXIS
                                      [0.000000, 0.000000, 0.000000, 1.000000]) # direction of robot
        for jointIndex in range(1,p.getNumJoints(self.baUid)): 
            p.resetJointState(self.baUid, jointIndex, jointPoses[jointIndex-1]) # 1,2,3,4,5,6,7
            p.setJointMotorControl2(bodyIndex=self.baUid,
                                jointIndex=jointIndex,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPoses[jointIndex-1],
                                targetVelocity=0,
                                force=500,
                                positionGain=0.03,
                                velocityGain=1)
    
    def applyAction(self, motorCommands): #4 actions
        limit_x=[-1,1]
        limit_y=[-1,1]
        limit_z=[0,1]
        def clip_val(val,limit):
            if val<limit[0]:
                return limit[0]
            if val>limit[1]:
                return limit[1]
            return val
        if (self.useInverseKinematics):

            dx = motorCommands[0]
            dy = motorCommands[1]
            dz = motorCommands[2]
            #da = motorCommands[3]
            fingerAngle = motorCommands[3] 
            state = p.getLinkState(self.baUid, self.baEndEffectorIndex) # returns 1. center of mass cartesian coordinates, 2. rotation around center of mass in quaternion
            actualEndEffectorPos = state[4] #world position of the link

            self.endEffectorPos[0] = clip_val(actualEndEffectorPos[0] + dx,limit_x)
            self.endEffectorPos[1] =  clip_val(actualEndEffectorPos[1] +  dy,limit_y)
            self.endEffectorPos[2] = clip_val(actualEndEffectorPos[2] +  dz,limit_z)
            
            pos = self.endEffectorPos
            orn = [0, 0, 0]
            jointPoses = inverse.getinversePoisition(self.baUid,pos)
            if (self.useSimulation):
                for i in range(1,self.baEndEffectorIndex+1): #1,2,3,4,5  
                    p.resetJointState(self.baUid, i, jointPoses[i-1]) 
                    p.setJointMotorControl2(bodyUniqueId=self.baUid,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=jointPoses[i-1],
                                  targetVelocity=0,
                                  force=self.maxForce,
                                  maxVelocity=self.maxVelocity,
                                  positionGain=0.3,
                                  velocityGain=1)
            self.sent_hand_moving(fingerAngle)
            
    def sent_hand_moving(self,motorCommand):
        left_hand_joint_now = p.getJointState(self.baUid,self.baFingerIndexL)[0]
        right_hand_joint_now = p.getJointState(self.baUid,self.baFingerIndexR)[0]
        p.setJointMotorControl2(self.baUid,  #control of fingers
                          self.baFingerIndexL,
                          p.POSITION_CONTROL,
                          targetPosition=motorCommand+left_hand_joint_now,
                          targetVelocity=0,
                          force=self.fingerTipForce)

        p.setJointMotorControl2(self.baUid,
                          self.baFingerIndexR,
                          p.POSITION_CONTROL,
                          targetPosition=motorCommand+right_hand_joint_now,
                          targetVelocity=0,
                          force=self.fingerTipForce)

    def grasping(self):
        # open gripper max angle
        p.setJointMotorControl2(self.baUid,
                          self.baFingerIndexL,
                          p.POSITION_CONTROL,
                          targetPosition=0.3490,
                          force=self.fingerTipForce)
        p.setJointMotorControl2(self.baUid,
                          self.baFingerIndexR,
                          p.POSITION_CONTROL,
                          targetPosition=0.3490,
                          force=self.fingerTipForce)

  

if __name__ == '__main__':
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    braccio_arm_test = braccio_arm_v0()
    p.setGravity(0,0,-10)
    #braccio_arm_test.applyAction([-0.5, -0.8, 0.2, 1.2570])
    #braccio_arm_test.grasping()
    #braccio_arm_test.get_to_place([0.0,0.5,0.0])
    for i in range (10000):
        p.stepSimulation()
        time.sleep(1./240.0)
    p.disconnect()
