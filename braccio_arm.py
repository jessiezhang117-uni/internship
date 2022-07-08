import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import pybullet as p
import time
import pybullet_data
import braccio_arm_inverse_kinematics as inverse
import math
import numpy as np

class braccio_arm_v0:

    def __init__(self, urdfRootPath=pybullet_data.getDataPath(), timeStep=0.01):
        self.urdfRootPath = urdfRootPath
        self.timeStep = timeStep
        self.maxVelocity = 1.0
        self.maxForce = 1000.
        self.fingerAForce = 10
        self.fingerBForce = 10
        self.useInverseKinematics = 1
        self.useSimulation = 1
        self.useNullSpace = 1
        self.useOrientation = 1
        self.baEndEffectorIndex = 5
        self.baFingerIndexL = 5
        self.baFingerIndexR = 6 #check urdf file with the index number
        # lower limits for null space
        self.ll = [0.0,0.2618,0.0,0.0,0.0,0.1750,0.1750]
        # upper limits for null space
        self.ul = [3.1416,2.8798,3.1416,3.1416,3.1416,1.2741,1.2741]
        # joint ranges for null space
        self.jr = [3.1415, 2.618, 3.1416, 3.1416, 3.1416, 1.0991, 1.0991]  
        # restposes for null space
        #self.rp = [0, 0, 0, 0.5 * math.pi, 0, -math.pi * 0.5 * 0.66, 0]
        self.rp =  [1.5708, 1.5708, 1.5708, 1.5708, 
        1.5708, 1.27409, 1.27409]
        # joint damping coefficents
        self.jd =  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.reset()

    def reset(self):
        #load model
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        planeId = p.loadURDF("plane.urdf")
        robot = p.loadURDF("/Users/jessiezhang/Documents/internship-1/model/braccio_arm_clean.urdf",useFixedBase=1) 
        # get barobot id
        self.baUid = robot
        # reset original position
        p.resetBasePositionAndOrientation(self.baUid, [0.0, 0.0, 0.0], # position of robot, GREEN IS Y AXIS
                                      [0.000000, 0.000000, 0.000000, 1.000000]) # direction of robot
        
        self.jointPositions = [  0.6,
    0.72402215,
    0.926779,
    1.0096856,
    0.20797996,
    0.4864421,
    0.4618157
  ]# 7 joints(including world joint) random reset joint angles
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
    
        self.endEffectorAngle = 1.5708 #0.02 #TODO

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
        state = p.getLinkState(self.baUid, self.baEndEffectorIndex)
        pos = state[0]
        orn = state[1]
        euler = p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(euler))

        return observation

    
    def applyAction(self, motorCommands): #4 actions
        if (self.useInverseKinematics):
            dx = motorCommands[0]
            dy = motorCommands[1]
            dz = motorCommands[2]
            da = motorCommands[3]
            #fingerAngle = motorCommands[3] 
            state = p.getLinkState(self.baUid, self.baEndEffectorIndex) # returns 1. center of mass cartesian coordinates, 2. rotation around center of mass in quaternion
            actualEndEffectorPos = state[0] #world position of the link

            self.endEffectorPos[0] = actualEndEffectorPos[0] + dx
            if (self.endEffectorPos[0] > 1.0):
                self.endEffectorPos[0] = 1.0
            if (self.endEffectorPos[0] < -1.0):
                self.endEffectorPos[0] = -1.0
            self.endEffectorPos[1] = self.endEffectorPos[1] + dy
            if (self.endEffectorPos[1] < -1.0):
                self.endEffectorPos[1] = -1.0
            if (self.endEffectorPos[1] > 1.0):
                 self.endEffectorPos[1] = 1.0
            self.endEffectorPos[2] = actualEndEffectorPos[2] + dz
            if (self.endEffectorPos[2] > 0.5):
                 self.endEffectorPos[2] = 0.5
            self.endEffectorAngle = self.endEffectorAngle + da
            if self.endEffectorAngle  >= np.pi:
              self.endEffectorAngle = np.pi
            if self.endEffectorAngle <= 0:
              self.endEffectorAngle = 0
     
            pos = self.endEffectorPos
            orn = [np.pi, 0, 0]
            if (self.useNullSpace == 1):
                if (self.useOrientation == 1):
                    jointPoses = list(p.calculateInverseKinematics(self.baUid, self.baEndEffectorIndex, pos,
                                                    orn, self.ll, self.ul, self.jr, self.rp))
                    jointPoses[4] = self.endEffectorAngle
                else:
                    jointPoses = p.calculateInverseKinematics(self.baUid,
                                                    self.baEndEffectorIndex,
                                                    pos,lowerLimits=self.ll,
                                                    upperLimits=self.ul,
                                                    jointRanges=self.jr,
                                                    restPoses=self.rp)
            else:
                if (self.useOrientation == 1):
                    jointPoses = list(p.calculateInverseKinematics(self.baUid,
                                                    self.baEndEffectorIndex,
                                                    pos,
                                                    orn,
                                                    jointDamping=self.jd))
                    jointPoses[4] = self.endEffectorAngle
                else:
                    jointPoses = p.calculateInverseKinematics(self.baUid, self.baEndEffectorIndex, pos)

            if (self.useSimulation):
                for i in range(self.baEndEffectorIndex): #0,1,2,3,4,5  
                    p.resetJointState(self.baUid, i, jointPoses[i]) 
                    p.setJointMotorControl2(bodyUniqueId=self.baUid,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=jointPoses[i],
                                  targetVelocity=0,
                                  force=self.maxForce,
                                  maxVelocity=self.maxVelocity,
                                  positionGain=0.3,
                                  velocityGain=1)
            # else:
            #     #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
            #     for i in range(self.numJoints):
            #         p.resetJointState(self.baUid,i,jointPoses[i])
            
            # p.setJointMotorControl2(self.baUid,
            #               self.baFingerIndexL,
            #               p.POSITION_CONTROL,
            #               targetPosition=fingerAngle,
            #               force=self.fingerAForce)

            # p.setJointMotorControl2(self.baUid,
            #               self.baFingerIndexR,
            #               p.POSITION_CONTROL,
            #               targetPosition=fingerAngle,
            #               force=self.fingerBForce)
        
        else:
            for action in range(len(motorCommands)):
                motor = self.motorIndices[action]
                p.setJointMotorControl2(self.baUid,
                                motor,
                                p.POSITION_CONTROL,
                                targetPosition=motorCommands[action],
                                force=self.maxForce)
            
            
    def grasping(self,action):
        # open gripper max angle
        p.setJointMotorControl2(self.baUid,
                          self.baFingerIndexL,
                          p.POSITION_CONTROL,
                          targetPosition=action,
                          force=self.fingerAForce)
        p.setJointMotorControl2(self.baUid,
                          self.baFingerIndexR,
                          p.POSITION_CONTROL,
                          targetPosition=action,
                          force=self.fingerBForce)
  

if __name__ == '__main__':
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    braccio_arm_test = braccio_arm_v0()
    #p.setGravity(0,0,-10)
    block =  p.loadURDF("/Users/jessiezhang/Documents/internship-1/model/cube.urdf",[0.2,0.2,0.02]) 
    
    #braccio_arm_test.applyAction([0.2, 0.2, -1,np.pi/4])
    #braccio_arm_test.grasping(1.0)
    # braccio_arm_test.applyAction([0.0,0.5,0.0,1.0])
    for i in range (10000):
        p.stepSimulation()
        time.sleep(1./240.0)
    p.disconnect()
