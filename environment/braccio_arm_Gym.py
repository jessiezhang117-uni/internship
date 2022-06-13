import os, inspect

import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.mujoco import mujoco_env

import numpy as np
import time
import mujoco_py as mp

import random
import braccio_arm

largeValObservation=100
RENDER_HEIGHT=720
RENDER_WIDTH=960
maxSteps=700
Dv=0.004

class braccio_arm_possensor_gym(gym.Env):
    metadata={'render.modes':['human','rgb_array'],'video.frames_per_secons':50}

    def __init__(self,
                urdfPath=mp.load_model_from_path("./model/braccio_arm.xml"),
                actionRepeat=1,
                isEnableSelfCollision=True,
                renders=False,
                isDiscrete=False,
                maxSteps=maxSteps):
        self._isDiscrete = isDiscrete
        self._timeStep = 1. / 240.
        self._urdfPath = urdfPath
        self._actionRepeat = actionRepeat
        self._isEnableSelfCollision = isEnableSelfCollision
        self._observation = []
        self._envStepCounter = 0
        self._renders = renders
        self._maxSteps = maxSteps
        self.terminated = 0
        self._cam_dist = 1.3
        self._cam_yaw = 180
        self._cam_pitch = -40

        self._mp = mp
        ###### to do: change to mujoco_py syntax
        if self._renders:
            cid=mp.connect(mp.SHARED_MEMORY)
            if(cid<0):
                cid=mp.connect(mp.GUI)
            mp.resetDebugVisualizerCamera(1.3,180,-41,[0.52,-0.2,-0.33])
        else:
            mp.connect(mp.DIRECT)

        self.seed()
        self.reset()
        observationDim=len(self.getExtendedObservation())

        observation_high=np.array([largeValObservation]*observationDim)
        if(self._isDiscrete):
            self.action_space =spaces.Discrete(7) #why 7?
        else:
            action_dim=3
            self._action_bound=1
            action_high=np.array([self._action_bound])*action_dim
            self.action_space=spaces.Box(-action_high,action_high)
        self.observation_space = spaces.Box(-observation_high,observation_high)
        self.viewer=None

    def reset(self):
        self.terminated=0
        # To do: change to mujoco_py syntax
        mp.resetSimulation()
        mp.setPhysicsEngineParameter(numSolverIterations=150)
        mp.setTimeStep(self._timeStep)
        mp.load_model_from_path("./model/braccio_arm.xml")

        # To do: add table urdf file
        self.tableUid = mp.loadURDF(os.path.join(self._urdfPath, "table/table.urdf"), 0.5000000, 0.00000, -.640000,
               0.000000, 0.000000, 0.0, 1.0)

        xpos = 0.55 +0.12*random.random()
        ypos = 0+0.2*random.random()
        ang =3.14*0.5+1.5#*random.random()
        orn = mp.getQuaternionFromEuler([0,0,ang])
        self.blockUid=mp.loadURDF(os.path.join(self._urdfRoot, "jenga/jenga.urdf"), xpos, ypos, 0.1,
                               orn[0], orn[1], orn[2], orn[3])
        mp.setGravity(0,0,-10)
        self._graccio_arm = graccio_arm.graccio_arm(urdfRootPath=self._urdfPath,timeStep=self._timeStep)
        self._envSteoCounter=0
        mp.stepSImulation()
        self._observation=self.getExtendedObservation()
        return np.array(self._observation)

    def __del__(self):
        mp.disconnect()

    def seed(self,seed=None):
        self.np_random,seed=seeding.np_random(seed)
        return [seed]

    def getExtendedObservation(self):
        self._observation=self._braccio_arm.getObservation()
        # to do 
        gripperState = p.getLinkState(self._tm700.tm700Uid, self._tm700.tmFingerIndexL)
        gripperStateR = p.getLinkState(self._tm700.tm700Uid, self._tm700.tmFingerIndexR)
            
        gripperPos = gripperState[0]
        gripperOrn = gripperState[1]
        gripperPosR = gripperStateR[0]
        gripperOrnR = gripperStateR[1]
        blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)

        invGripperPos, invGripperOrn = p.invertTransform(gripperPos, gripperOrn)
        invGripperPosR, invGripperOrnR = p.invertTransform(gripperPosR, gripperOrnR)

        gripperMat = p.getMatrixFromQuaternion(gripperOrn)
        gripperMatR = p.getMatrixFromQuaternion(gripperOrnR)

        blockPosInGripper, blockOrnInGripper = p.multiplyTransforms(invGripperPos, invGripperOrn,
                                                                blockPos, blockOrn)
        blockPosInGripperR, blockOrnInGripperR = p.multiplyTransforms(invGripperPosR, invGripperOrnR,
                                                                blockPos, blockOrn)
        blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
        blockEulerInGripperR = p.getEulerFromQuaternion(blockOrnInGripperR)

        #we return the relative x,y position and euler angle of block in gripper space
        blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]
        blockInGripperPosXYEulZR = [blockPosInGripperR[0], blockPosInGripperR[1], blockEulerInGripper[2]]

        self._observation.extend(list(blockInGripperPosXYEulZ))
        self._observation.extend(list(blockInGripperPosXYEulZR))

        return self._observation
    
    # what are these numbers?
    def step(self,action):
        if(self._isDiscrete):
            dv=Dv
            dx = [0, -dv, dv, 0, 0, 0, 0][action]
            dy = [0, 0, 0, -dv, dv, 0, 0][action]
            da = [0, 0, 0, 0, 0, -0.05, 0.05][action]
            f=0.15
            realAction=[dx,dy,-0.0005,da,f]
        else:
            dv = Dv
            dx = action[0] * dv
            dy = action[1] * dv
            da = action[2] * 0.05
            f = 0.15
            realAction = [dx, dy, -0.0005, da, f]
        return self.step2(realAction)

    def step2(self,action):
        for i in range(self._actionRepeat):
            self._braccio_arm.applyAction(action)
            mp.stepSimulation()
            if self._termination():
                break
            self._envStepCounter+=1
            if self._renders:
                time.sleep(self._timeStep)
            self._observation=self.getExtendedObservation()

        done = self._termination()
        npaction=np.array([
                action[3]
            ])#only penalize rotation until learning works well
        actionCost=np.linalg.norm(npaction)*10.
        reward=self._reward()-actionCost
        return np.array(self._observation),reward,done,{}

    def render(self,mode='rgb_array',close=False):
        if mode!= 'rgb_array':
            return np.array([])
        # to do : mujoco_py
        base_pos,orn = self._mp.getBasePositionAndOrientation(self._braccio_arm.braccio_armUid)
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
        # to do : mujoco_py
        state = mp.getLinkState(self._braccio_arm.braccio_armUid, self._braccio_arm.baEndEffectorIndex)
        actualEndEffectorPos = state[0]
        if (self.terminated or self._envStepCounter>self._maxSteps):
            self._observation=self.getExtendedObservation()
            return True
        # to do
        maxDist=0.006
        closestPoints=mp.getClosestPoints(self.tableUid, self._tm700.tm700Uid, maxDist, -1, self._tm700.tmFingerIndexL)

        if (len(closestPoints)):
            self.terminated=1
            #start grasp and terminate
            fingerAngle=0.15
            for i in range(1000):
                graspAction=[0,0,0.0005,0,fingerAngle]
                mp.stepSimulation()
                fingerAngle=fingerAngle-(0.3/100.)
                if (fingerAngle<0):
                    fingerAngle=0
            # To do
            for i in range(10000):
                graspAction=[0,0,0.001,0,fingerAngle]
                self._graccio_arm.applyAction(graspAction)
                mp.stepsimulation()
                blockPos, blockOrn = mp.getBasePositionAndOrientation(self.blockUid)
                if (blockPos[2] > 0.23):
                    break
                state = mp.getLinkState(self._tm700.tm700Uid, self._tm700.tmEndEffectorIndex)
                actualEndEffectorPos = state[0]
                if (actualEndEffectorPos[2] > 0.5):
                    break

            self._observation = self.getExtendedObservation()
            return True
        return False

def _reward(self):
    # To do
    #rewards is height of target object
    blockPos, blockOrn = p.getBasePositionAndOrientation(self.blockUid)
    closestPoints1 = p.getClosestPoints(self.blockUid, self._tm700.tm700Uid, 10, -1,
                                       self._tm700.tmFingerIndexL)
    closestPoints2 = p.getClosestPoints(self.blockUid, self._tm700.tm700Uid, 10, -1,
                                       self._tm700.tmFingerIndexR) # id of object a, id of object b, max. separation, link index of object a (base is -1), linkindex of object b

    # fingerL = p.getLinkState(self._tm700.tm700Uid, self._tm700.tmFingerIndexL)
    # fingerR = p.getLinkState(self._tm700.tm700Uid, self._tm700.tmFingerIndexR)
    # print('infi', np.mean(list(fingerL[0])))

    reward =-1000
    closestPoints = closestPoints1[0][8]
    numPt = len(closestPoints1)
    if (numPt > 0):
      # reward = -1./((1.-closestPoints1[0][8] * 100 + 1. -closestPoints2[0][8] * 100 )/2)
      reward = -((closestPoints1[0][8])**2 + (closestPoints2[0][8])**2 )*(1/0.17849278457978357)
      # reward = 1/((abs(closestPoints1[0][8])   + abs(closestPoints2[0][8])*10 )**2 / 2)
      # reward = 1/closestPoints1[0][8]+1/closestPoints2[0][8]
    if (blockPos[2] > 0.2):
      reward = reward + 1000
      print("successfully grasped a block!!!")
    return reward

if __name__ == '__main__':

# datapath = pybullet_data.getDataPath()
  mp.connect(mp.GUI, options="--opencl2")
  #p.setAdditionalSearchPath(datapath)
  test =braccio_arm_possensor_gym()
  for i in range(10000):
    # test.step2([0.55, 0.2, 0.05,0,0])
    mp.stepSimulation()
    # tm700test.print_joint_state()
    time.sleep(1. / 240.0)

  time.sleep(50)   



