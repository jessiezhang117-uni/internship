import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import random
import pybullet_data
import braccio_arm_test

largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960


class BraccioArmGymEnv(gym.Env):
  metadata = {'render.modes': ['human', 'rgb_array'], 'video.frames_per_second': 50}

  def __init__(self,
               urdfRoot=pybullet_data.getDataPath(),
               actionRepeat=1,
               isEnableSelfCollision=True,
               renders=False,
               isDiscrete=False,
               maxSteps=10000):
    self._isDiscrete = isDiscrete
    self._timeStep = 1. / 240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    
    self._renders = renders
 
    self._cam_dist = 1.3
    self._cam_yaw = 90
    self._cam_pitch = -45

    self._envStepCounter = 0
    self.terminated = 0
    self._attepmted_grasp = False

    self._maxSteps = maxSteps
    
    self._graspSuccess = 0
    self._totalGraspTimes = 0

    self._p = p
    if self._renders:
      cid = p.connect(p.SHARED_MEMORY)
      if (cid < 0):
        cid = p.connect(p.GUI)
      p.resetDebugVisualizerCamera(1.3, 90, -45, [0, 0, 0])
    else:
      p.connect(p.DIRECT)
    self.seed()
    self.reset()
    observationDim = len(self.getExtendedObservation())
    #print("observationDim")
    #print(observationDim)

    observation_high = np.array([largeValObservation] * observationDim)
    if (self._isDiscrete):
      self.action_space = spaces.Discrete(7)
    else:
      action_dim = 4
      self._action_bound = 1
      action_high = np.array([self._action_bound] * action_dim)
      self.action_space = spaces.Box(np.float32(-action_high), np.float32(action_high),dtype=np.float32)
    self.observation_space = spaces.Box(np.float32(-observation_high), np.float32(observation_high),dtype=np.float32)
    self.viewer = None

  def reset(self):
    #print("KukaGymEnv _reset")
    self.terminated = 0
    p.resetSimulation()
    p.setPhysicsEngineParameter(numSolverIterations=150)
    p.setTimeStep(self._timeStep)
    p.setGravity(0, 0, -10)
    self._attepted_grasp = False
    self._envStepCounter = 0
    self.terminated = 0
    self._ba = braccio_arm_test.BraccioArm(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    xpos = 2*0.3*random.random() -0.2
    ypos = 2*0.3*random.random() -0.2
    zpos = 0.02 #fix z-axis position
    ang = np.pi/2  + np.pi * random.random()
    orn = p.getQuaternionFromEuler([0, 0, ang])
    self.blockUid = p.loadURDF("/home/jessie/internship/model/cube.urdf",basePosition=[xpos, ypos, zpos],
                               baseOrientation=[orn[0], orn[1], orn[2], orn[3]],useFixedBase=1)
    p.stepSimulation()
    self._observation = self.getExtendedObservation()
    return np.array(self._observation).astype("float32")

  def __del__(self):
    p.disconnect()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):
    self._observation = self._ba.getObservation()
    gripperState = p.getLinkState(self._ba.baUid, self._ba.baGripperIndex)
    gripperPos = gripperState[0]
    gripperOrn = gripperState[1]
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)

    invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
    gripperMat = p.getMatrixFromQuaternion(gripperOrn)
    dir0 = [gripperMat[0], gripperMat[3], gripperMat[6]]
    dir1 = [gripperMat[1], gripperMat[4], gripperMat[7]]
    dir2 = [gripperMat[2], gripperMat[5], gripperMat[8]]

    gripperEul = p.getEulerFromQuaternion(gripperOrn)
    #print("gripperEul")
    #print(gripperEul)
    blockPosInGripper, blockOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn,
                                                                blockPos, blockOrn)
    projectedBlockPos2D = [blockPosInGripper[0], blockPosInGripper[1]]
    blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)

    #we return the relative x,y position and euler angle of block in gripper space
    blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]

    self._observation.extend(list(blockInGripperPosXYEulZ))
    return self._observation

  def step(self, action):
    if (self._isDiscrete):
      dv = 0.15 #velocity per physics step
      dx = [0, -dv, dv, 0, 0, 0, 0][action]
      dy = [0, 0, 0, -dv, dv, 0, 0][action]
      dz = [0, 0, 0, 0, -dv, dv, 0][action]
      da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
      f = 1
      realAction = [dx, dy, dz , da,f]
    else:
      dv = 0.15
      dx = action[0]*dv
      dy = action[1]*dv
      dz = action[2]*dv
      da = action[3]*0.25
      f = 1
      realAction = [dx, dy, dz , da, f]
    return self.step2(realAction)

  def step2(self, action):
    # perform commanded action
    self._envStepCounter += 1
    self._ba.applyAction(action)
    for _ in range(self._actionRepeat):
      p.stepSimulation()
      if self._renders:
        time.sleep(self._timeStep)
      if self._termination():
        break
    # if we are close to the block, attempt grasp
    state = p.getLinkState(self._ba.baUid, self._ba.baEndEffectorIndex)
    actualEndEffectorPos = state[0]
    maxDist = 0.005
    closestPoints = p.getClosestPoints(self.blockUid, self._ba.baUid, maxDist)
    if (len(closestPoints)): 
      #start grasp and terminate
      fingerAngle = 1.0
      for i in range(500):
        graspAction = [0, 0, 0, 0, fingerAngle]
        self._ba.applyAction(graspAction)
        p.stepSimulation()
        fingerAngle = fingerAngle - (1.0 / 100.)
        if (fingerAngle < 0):
          fingerAngle = 0
      for i in range(500):
        graspAction = [0, 0, 0.001, 0, fingerAngle]
        self._ba.applyAction(graspAction)
        p.stepSimulation()
        if self._renders:
          time.sleep(self._timeStep)
        fingerAngle -=1.0/100.0
        if fingerAngle<0:
          fingerAngle=0
      self._attempted_grasp=True
    self._observation = self.getExtendedObservation()
    done = self._termination()
    npaction = np.array([
        action[3]
    ])  #only penalize rotation until learning works well [action[0],action[1],action[3]])
    actionCost = np.linalg.norm(npaction) * 10.
    reward = self._reward() - actionCost
    if reward>1000:
      self._graspSuccess +=1
    info = {"grasp_success":self._graspSuccess}
    if done:
      self._totalGraspTimes+=1
      print(
                f"done: {done}, reward: {reward}, success_grasp_times: {self._graspSuccess}, total_grasp_times: {self._totalGraspTimes}")
    return np.array(self._observation), reward, done, info

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])

    base_pos, orn = self._p.getBasePositionAndOrientation(self._ba.baUid)
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
                                              renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
    #renderer=self._p.ER_TINY_RENDERER)

    rgb_array = np.array(px, dtype=np.uint8)
    rgb_array = np.reshape(rgb_array, (RENDER_HEIGHT, RENDER_WIDTH, 4))

    rgb_array = rgb_array[:, :, :3]
    return rgb_array

  def _termination(self):
    if (self.terminated or self._envStepCounter > self._maxSteps):
      self._observation = self.getExtendedObservation()
      return True
    return False

  def _reward(self):
    reward = 0
    self._graspSuccess=0
    #rewards is height of target object
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    # if block is above height, provide reward
    if (blockPos[2] > 0.2):
      self._graspSuccess +=1
      reward = 1000
      #print("successfully grasped a block!!!")
    elif self._envStepCounter > self._maxSteps:
      reward =-0.1
    return reward

if __name__ == '__main__':
    env = BraccioArmGymEnv(renders=True, isDiscrete=False) 
    for i in range (10000):
        p.stepSimulation()
        time.sleep(1./240.0)