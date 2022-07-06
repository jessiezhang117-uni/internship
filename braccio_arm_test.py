import pybullet as p
import numpy as np
import copy
import math
import pybullet_data


class BraccioArm:

  def __init__(self, urdfRootPath=pybullet_data.getDataPath(),timeStep=1.0/240,basePosition=[0, 0, 0],EndEffectorPosition=[0, 0, 0.3],gripperAngle=0):
    self.urdfRootPath = urdfRootPath
    self.timeStep = timeStep
    self._basePosition = basePosition 
    self._baEndEffectorPosition = EndEffectorPosition 
    self._gripperAngle = gripperAngle 
    self.maxVelocity = 1.0
    self.maxForce = 1000.
    self.fingerAForce = 10
    self.fingerBForce = 10
    self.useInverseKinematics = 1
    self.useSimulation = 1
    self.useNullSpace = 21
    self.useOrientation = 1
    self.baEndEffectorIndex = 5
    self.baGripperIndex = 5
    #lower limits for null space
    self.ll = [0.0,0.2618,0.0,0.0,0.0,0.1750,0.1750]
    #upper limits for null space
    self.ul = [3.1416,2.8798,3.1416,3.1416,3.1416,1.2741,1.2741]
    #joint ranges for null space
    self.jr = [3.1415, 2.618, 3.1416, 3.1416, 3.1416, 1.0991, 1.0991]
    #restposes for null space
    self.rp =  [1.5708, 1.5708, 1.5708, 1.5708, 
        1.5708, 1.27409, 1.27409]
    #joint damping coefficents
    self.jd = [
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    if self.useOrientation:
      # gripper always looking down
      self._euler = [np.pi, 0, 0]
      self.quaternion = p.getQuaternionFromEuler(self._euler)
    self.reset()

  def reset(self):
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    planeId = p.loadURDF("plane.urdf")
    robot = p.loadURDF("/home/jessie/internship/model/braccio_arm_clean.urdf") 
    self.baUid = robot
    p.resetBasePositionAndOrientation(self.baUid, [0.0, 0.0, 0.0], # position of robot, GREEN IS Y AXIS
                                      [0.000000, 0.000000, 0.000000, 1.000000]) # direction of robot
    #for i in range (p.getNumJoints(self.kukaUid)):
    #  print(p.getJointInfo(self.kukaUid,i))
    self.setOriginalPosition()
    self.numJoints = p.getNumJoints(self.baUid) 
    self.motorNames = []
    self.motorIndices = []
   

    for i in range(self.numJoints):
      jointInfo = p.getJointInfo(self.baUid, i)
      qIndex = jointInfo[3]
      if qIndex > -1:
        self.motorNames.append(str(jointInfo[1]))
        self.motorIndices.append(i)

  def getObservation(self):
    observation = []
    state = p.getLinkState(self.baUid, self.baEndEffectorIndex)
    pos = state[4]
    orn = state[5]
    euler = p.getEulerFromQuaternion(orn)

    observation.extend(list(pos))
    observation.extend(list(euler))

    return observation

  def applyAction(self, motorCommands):
    #print ("self.numJoints")
    #print (self.numJoints)
    if (self.useInverseKinematics):

      dx = motorCommands[0]
      dy = motorCommands[1]
      dz = motorCommands[2]
      da = motorCommands[3]
      fingerAngle = motorCommands[4]

      state = p.getLinkState(self.baUid, self.baEndEffectorIndex)
      actualEndEffectorPos= state[0]

      self._baEndEffectorPosition[0] = actualEndEffectorPos[0] + dx
      if (self._baEndEffectorPosition[0] > 5):
        self._baEndEffectorPosition[0] = 5
      if (self._baEndEffectorPosition[0] < -5):
        self._baEndEffectorPosition[0] = -5
      self._baEndEffectorPosition[1] = actualEndEffectorPos[1] + dy
      if (self._baEndEffectorPosition[1] < -5):
        self._baEndEffectorPosition[1] = -5
      if (self._baEndEffectorPosition[1] > 5):
        self._baEndEffectorPosition[1] = 5
      self._baEndEffectorPosition[2] = actualEndEffectorPos[2] + dz
      if (self._baEndEffectorPosition[2]<0.0):
        self._baEndEffectorPosition[2]=0.0
    
      self._gripperAngle = self._gripperAngle + da
      if self._gripperAngle >= np.pi*2:
        self._gripperAngle = np.pi*2
      if self._gripperAngle <= -np.pi*2:
        self._gripperAngle =-np.pi*2
      pos = self._baEndEffectorPosition
      orn = p.getQuaternionFromEuler([np.pi, 0, 0])  # -math.pi,yaw]) #gripper keeps looking down
      if (self.useNullSpace == 1):
        if (self.useOrientation == 1):
          jointPoses = list(p.calculateInverseKinematics(self.baUid, self.baEndEffectorIndex, pos,
                                                    orn, self.ll, self.ul, self.jr, self.rp))
          jointPoses[5] = self._gripperAngle
        else:
          jointPoses = p.calculateInverseKinematics(self.baUid,
                                                    self.baEndEffectorIndex,
                                                    pos,
                                                    lowerLimits=self.ll,
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
          jointPoses[5] = self._gripperAngle
        else:
          jointPoses = p.calculateInverseKinematics(self.baUid, self.baEndEffectorIndex, pos)
      if (self.useSimulation):
        for i in range(1,5):
          #print(i)
          p.setJointMotorControl2(bodyUniqueId=self.baUid,
                                  jointIndex=i,
                                  controlMode=p.POSITION_CONTROL,
                                  targetPosition=jointPoses[i-1],
                                  targetVelocity=0,
                                  force=self.maxForce,
                                  maxVelocity=self.maxVelocity,
                                  positionGain=0.3,
                                  velocityGain=1)
      else:
        #reset the joint state (ignoring all dynamics, not recommended to use during simulation)
        for i in range(self.numJoints):
          p.resetJointState(self.baUid, i, jointPoses[i-1])
      #fingers
      # p.setJointMotorControl2(self.baUid,
      #                         5,
      #                         p.POSITION_CONTROL,
      #                         targetPosition=self._gripperAngle,
      #                         force=self.maxForce)
      p.setJointMotorControl2(self.baUid,
                              6,
                              p.POSITION_CONTROL,
                              targetPosition=fingerAngle,
                              force=self.fingerAForce)
      p.setJointMotorControl2(self.baUid,
                              7,
                              p.POSITION_CONTROL,
                              targetPosition=fingerAngle,
                              force=self.fingerBForce)

    else:
      for action in range(len(motorCommands)):
        motor = self.motorIndices[action]
        p.setJointMotorControl2(self.baUid,
                                motor,
                                p.POSITION_CONTROL,
                                targetPosition=motorCommands[action],
                                force=self.maxForce)

  def setOriginalPosition(self):
    if (self.useOrientation == 1):
      original_pose = list(p.calculateInverseKinematics(self.baUid,
                                                    self.baEndEffectorIndex,
                                                    self._baEndEffectorPosition,
                                                    self.quaternion,
                                                    jointDamping=self.jd))
      original_pose[5] =0 
    else:
        original_pose =p.calculateInverseKinematics(
                self.baUid, self.baEndEffectorIndex, self._baEndEffectorPosition
            )
    for i in range(1,5):
      p.setJointMotorControl2(
                    bodyIndex=self.baUid,
                    jointIndex=i,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=original_pose[i - 1],
                    force=self.maxForce,
                )
    p.setJointMotorControl2(self.baUid,
                              6,
                              p.POSITION_CONTROL,
                              targetPosition=1.5,
                              force=self.fingerAForce)
    p.setJointMotorControl2(self.baUid,
                              7,
                              p.POSITION_CONTROL,
                              targetPosition=1.5,
                              force=self.fingerBForce)
import time

if __name__ == '__main__':
    physicsClient = p.connect(p.GUI)#or p.DIRECT for non-graphical version
    braccio_arm_test = BraccioArm()
    p.setGravity(0,0,-10)
    #braccio_arm_test.grasping()
    braccio_arm_test.applyAction([5,5,2.5,np.pi/2,1])
    for i in range (10000):
        p.stepSimulation()
        time.sleep(1./240.0)
    p.disconnect()