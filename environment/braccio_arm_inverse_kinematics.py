import pybullet as p
import time
import math
from datetime import datetime
import pybullet_data

def getinversePoisition(baUid,position_desired,orientation_desired=[]):
    joints_info = []
    joint_damping = []
    joint_ll = [0.0,0.2618,0.0,0.0,0.0,0.1750,0.1750]
    joint_ul = [3.1416,2.8798,3.1416,3.1416,3.1416,1.2741,1.2741]
    useOrientation=len(orientation_desired)
    for i in range(p.getNumJoints(baUid)):
        joints_info.append(p.getJointInfo(baUid, i))
    baEndEffectorIndex = 5 
    numJoints = p.getNumJoints(baUid)
    useNullSpace = 1
    ikSolver = 1
    pos = [position_desired[0], position_desired[1], position_desired[2]]
    #end effector points down, not up (in case useOrientation==1)
    if useOrientation:
        orn = p.getQuaternionFromEuler([orientation_desired[0],orientation_desired[1] , orientation_desired[2]])
    if (useNullSpace == 1):
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(baUid, baEndEffectorIndex, pos, orn)
      else:
        jointPoses = p.calculateInverseKinematics(baUid,
                                                  baEndEffectorIndex,
                                                  pos,
                                                  lowerLimits=joint_ll,
                                                  upperLimits=joint_ul,
                                               )
    else:
      if (useOrientation == 1):
        jointPoses = p.calculateInverseKinematics(baUid,
                                                  baEndEffectorIndex,
                                                  pos,
                                                  orn,
                                                  solver=ikSolver,
                                                  maxNumIterations=100,
                                                  residualThreshold=.01)
      else:
        jointPoses = p.calculateInverseKinematics(baUid,
                                                  baEndEffectorIndex,
                                                  pos,
                                                  solver=ikSolver)
    return jointPoses