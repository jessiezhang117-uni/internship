import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
import time
import pybullet as p
import random
import pybullet_data
import braccio_arm

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
               maxSteps=1000):
    self._isDiscrete = isDiscrete
    self._timeStep = 1. / 240.
    self._urdfRoot = urdfRoot
    self._actionRepeat = actionRepeat
    self._isEnableSelfCollision = isEnableSelfCollision
    self._observation = []
    
    self._renders = renders
 
    self._cam_dist = 1.0
    self._cam_yaw = 50
    self._cam_pitch = -35

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
      p.resetDebugVisualizerCamera(1,50, -35, [-0.04, 0.03, -0.04])
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
    self.control_time = 1/20
    self._ba = braccio_arm.braccio_arm_v0(urdfRootPath=self._urdfRoot, timeStep=self._timeStep)
    xpos = 0.2
    ypos = 0.2
    zpos = 0.02
    # ang = np.pi/2  + np.pi * random.random()
    # orn = p.getQuaternionFromEuler([0, 0, ang])
    self.blockUid = p.loadURDF("/content/drive/MyDrive/internship-main/model/cube.urdf", basePosition=[xpos, ypos, zpos])
    for _ in range(int(self.control_time/self._timeStep)*10):
      p.stepSimulation()

    cube_pose,_ = p.getBasePositionAndOrientation(self.blockUid)
    self.cube_init_z = cube_pose[2]

    self._observation = self.getExtendedObservation()
    return np.array(self._observation).astype("float32")

  def __del__(self):
    p.disconnect()

  def seed(self, seed=None):
    self.np_random, seed = seeding.np_random(seed)
    return [seed]

  def getExtendedObservation(self):
    self._observation = self._ba.getObservation()
    gripperState = p.getLinkState(self._ba.baUid, self._ba.baFingerIndexL)
    gripperStateR = p.getLinkState(self._ba.baUid, self._ba.baFingerIndexR)
    
    gripperPos = gripperState[0]
    gripperOrn = gripperState[1]
    gripperPosR = gripperStateR[0]
    gripperOrnR = gripperStateR[1]

    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)

    invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
    invGripperPosR, invGripperOrnR = p.invertTransform(gripperPosR, gripperOrnR)

    gripperMat = p.getMatrixFromQuaternion(gripperOrn)
    gripperMat = p.getMatrixFromQuaternion(gripperOrnR)

    blockPosInGripper, blockOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn,
                                                                blockPos, blockOrn)
    blockPosInGripperR, blockOrnInGripperR = p.multiplyTransforms(invGripperPosR, invGripperOrnR,
                                                                blockPos, blockOrn)
    blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
    blockEulerInGripperR = p.getEulerFromQuaternion(blockOrnInGripperR)

    #we return the relative x,y position and euler angle of block in gripper space
    blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]
    blockInGripperPosXYEulZR = [blockPosInGripperR[0], blockPosInGripperR[1], blockEulerInGripperR[2]]

    self._observation.extend(list(blockInGripperPosXYEulZ))
    self._observation.extend(list(blockInGripperPosXYEulZR))
    return self._observation

  def step(self, action):
    dv = 0.004
    dx = action[0] *dv
    dy = action[1] *dv
    dz = action[2] *dv
    #da = action[3] *0.05
    f = action[3]
    realAction = [dx, dy ,dz, f]
    return self.step2(realAction)

  def step2(self, action):
    # perform commanded action
    for i in range(self._actionRepeat):
      self._ba.applyAction(action[0:3])
      p.stepSimulation()
      if self._termination():
        break
      self._envStepCounter+=1
    if self._renders:
      time.sleep(self._timeStep)
    # update obs
    obs = self.getExtendedObservation()

    # update termination
    done = self._termination()

    #update reward
    reward = self._reward()

    #update info
    info = {"grasp_success":self._graspSuccess}
    if done:
      self._totalGraspTimes+=1
      print(
                f"done: {done}, reward: {reward}, success_grasp_times: {self._graspSuccess}, total_grasp_times: {self._totalGraspTimes}")
    return np.array(self._observation), reward, done, info

  def render(self, mode="rgb_array", close=False):
    if mode != "rgb_array":
      return np.array([])

    #base_pos, orn = self._p.getBasePositionAndOrientation(self._ba.baUid)
    view_matrix = self._p.computeViewMatrixFromYawPitchRoll(cameraTargetPosition=[-0.04, 0.03, -0.04],
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
    # if we are close to the block, attempt grasp
    state = p.getLinkState(self._ba.baUid, self._ba.baEndEffectorIndex)
    actualEndEffectorPos = state[0]

    if (self.terminated or self._envStepCounter > self._maxSteps):
      self._observation = self.getExtendedObservation()
      return True
    maxDist = 0.005
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    closestPoints = p.getClosestPoints(self.blockUid, self._ba.baUid, maxDist, -1,self._ba.baFingerIndexL)
    
    if len(closestPoints):
      self.terminated =1          
      fingerAngle = 1.0
      for i in range(1000):
        #graspAction = [0,0,0,fingerAngle]
        #self._ba.applyAction(graspAction)
        self._ba.grasping(fingerAngle)
        p.stepSimulation()
        fingerAngle = fingerAngle - (1.0 / 100.)
        if (fingerAngle < 0):
          fingerAngle = 0
      for i in range(10000):
        action = [0,0,0.001]
        self._ba.applyAction(action)
        self._ba.grasping(fingerAngle)
        p.stepSimulation()
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
        if (blockPos[2]>0.33):
          break
        state = p.getLinkState(self._ba.baUid, self._ba.baEndEffectorIndex)
        actualEndEffectorPos = state[0]
        if (actualEndEffectorPos[2] > 0.5):
          break
      self._attepmted_grasp = True
      return True
    return False

  def _reward(self):
    reward=0
    self._graspSuccess=0
    
    # reaching reward
    # xy reward 

    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    gripper_pose = list(p.getLinkState(self._ba.baUid,self._ba.baEndEffectorIndex)[0])
    #dist = math.hypot( (gripper_pose[0] - blockPos[0]), (blockPos[1] - gripper_pose[1]) )
    dist = np.linalg.norm(np.array(gripper_pose)-np.array(blockPos))
    # reaching_reward = 1-np.tanh(dist)
    # reward +=reaching_reward

    # z reward
    distz = abs(gripper_pose[2]-blockPos[2])
    reward += 1-np.tanh(distz)

    # grasped reward 
    closestPoints1 = p.getClosestPoints(self.blockUid, self._ba.baUid, 0.005, -1,
                                       self._ba.baFingerIndexL)
    closestPoints2 = p.getClosestPoints(self.blockUid, self._ba.baUid, 0.005, -1,
                                       self._ba.baFingerIndexR)
    if closestPoints1!=() and closestPoints2!=():
      reward +=5


    # lift reward 
    if self._attepmted_grasp and blockPos[2]-self.cube_init_z>0.03:
      self._graspSuccess +=1
      reward += 100
      print("successfully grasped a block!!!")
    
    # punish 
    if self._envStepCounter > self._maxSteps:
      reward -=0.1

    return reward

if __name__ == '__main__':
    env = BraccioArmGymEnv(renders=True, isDiscrete=False) 
    for i in range (10000):
        p.stepSimulation()
        time.sleep(1./240.0)