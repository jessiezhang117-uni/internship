import numpy as np
import gym
import os
import math
import pybullet as p
import braccio_arm
from gym.utils import seeding
import pybullet_data
import random
import  time
from gym import spaces
from arguments import Args

largeValObservation = 100

# RENDER_HEIGHT = 720
# RENDER_WIDTH = 960

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class barobotGymEnv(gym.Env):
    def __init__(
        self, 
        n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold,reward_type
    ):
        """Initializes a new Fetch environment.

        Args:
            model_path (string): XML文件的路径，这里可以写URDF，在bmirobot里用的是Pybullet环境
            n_substeps (int): 目前推测n-substep是 每次step用的步数。比如一个动作发出后，后续25个时间步骤就继续执行动作
            gripper_extra_height (float): 当定位夹持器时，额外的高度高于桌子
            block_gripper (boolean): 抓手是否被阻塞(即不能移动)
            has_object (boolean):环境中是否有对象
            target_in_the_air (boolean):目标物是否应该在桌子上方的空中或桌面上
            target_offset (float or array with 3 elements): 目标偏移量
            obj_range (float): 初始目标位置采样的均匀分布范围
            target_range (float):采样目标的均匀分布范围
            distance_threshold (float): 目标达到之后的临界值
            initial_qpos (dict):定义初始配置的联合名称和值的字典
            reward_type ('sparse' or 'dense'):奖励类型，如稀疏或密集
        """
        IS_USEGUI = Args().Use_GUI
        self.gripper_extra_height = gripper_extra_height
        self.block_gripper = block_gripper
        self.has_object = has_object
        self.target_in_the_air = target_in_the_air
        self.target_offset = target_offset
        self.obj_range = obj_range
        self.target_range = target_range
        self.distance_threshold = distance_threshold
        self.reward_type = reward_type
        self.n_substeps=n_substeps
        self.n_actions=4
        self.blockUid = -1
        # self.initial_qpos=initial_qpos
        self._urdfRoot = pybullet_data.getDataPath()
        
        if IS_USEGUI:
            self.physics = p.connect(p.GUI)
        else:
            self.physics = p.connect(p.DIRECT)

        self.seed()
        # observationDim = len(self._get_obs())
        # observation_high = np.array([largeValObservation] * observationDim)
        #加载机器人模型
        self._barobot = braccio_arm.braccio_arm_v0()
        self._timeStep= 1. / 240.
        action_dim = 4
        self._action_bound = 0.5
        # 这里的action和obs space 的low and high 可能需要再次考虑
        action_high = np.array([self._action_bound] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high)
        #self.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float32)
        # self.observation_space = spaces.Box(-observation_high, observation_high)
        #重置环境
        # self.reset()
        obs = self.reset()  # required for init; seed can be changed later
        observationDim = len(self._get_obs())
        observation_high = np.array([largeValObservation] * observationDim)
        self.observation_space = spaces.Box(-observation_high, observation_high)

        # achieved_goal_shape = obs["achieved_goal"].shape
        # desired_goal_shape = obs["achieved_goal"].shape
        # self.observation_space = gym.spaces.Dict(
        #     dict(
        #         observation=gym.spaces.Box(-np.inf, np.inf, shape=observation_shape, dtype=np.float32),
        #         desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=achieved_goal_shape, dtype=np.float32),
        #         achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=desired_goal_shape, dtype=np.float32),
        #     )
        # )
        # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32)
        else:
            return -d

    def step(self, action):
        action = np.clip(action,-0.5,0.5)
        if p.getCLosetPoints(self._barobot.baUid,self.blockUid,0.0001):
            action[3]=-1
        self._set_action(action)
        # if (p.getClosestPoints(self._bmirobot.bmirobotid, self.blockUid, 0.0001)): #如果臂和块足够靠近，可以锁死手爪
        #     action[3]=-1
        # print(action[3])
        #一个动作执行20个仿真步
        for _ in range(self.n_substeps):
            p.stepSimulation()
        obs = self._get_obs()
        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        return obs, reward, done, info

    def reset(self):
        p.setPhysicsEngineParameter(numSolverIterations=150)
        # 选择约束求解器迭代的最大次数。如果达到了solverResidualThreshold，求解器可能会在numsolver迭代之前终止
        for i in range(8):
            p.resetJointState(self._barobot.baUid, i, 0, 0)
        p.setTimeStep(self._timeStep)
        # Cube Pos
        for _ in range(100):
            xpos = 0.05 +0.2 * random.random()  # 0.35
            ypos = (random.random() * 0.03) + 0.2  # 0.10 0.50
            zpos = 0.2
            ang = 3.14 * 0.5 + 3.1415925438 * random.random()
            orn = p.getQuaternionFromEuler([0, 0, ang])
            # target Position：
            xpos_target = 0.35 * random.random()  # 0.35
            ypos_target = (random.random() * 0.03) + 0.2  # 0.10 0.50
            zpos_target = 0.2
            ang_target = 3.14 * 0.5 + 3.1415925438 * random.random()
            orn_target = p.getQuaternionFromEuler([0, 0, ang_target])
            self.dis_between_target_block = math.sqrt(
                (xpos - xpos_target) ** 2 + (ypos - ypos_target) ** 2 + (zpos - zpos_target) ** 2)
            if self.dis_between_target_block >= 0.1:
                break
        if self.blockUid == -1:
            self.blockUid = p.loadURDF("/home/jessie/internship/model/cube.urdf", xpos, ypos, zpos,
                                       orn[0], orn[1], orn[2], orn[3])
            self.targetUid = p.loadURDF("/home/jessie/internship/model/cube_target.urdf",
                                        [xpos_target, ypos_target, zpos_target],
                                        orn_target, useFixedBase=1)
        # else:
        #     p.removeBody(self.blockUid)
        #     p.removeBody(self.targetUid)
        #     self.blockUid = p.loadURDF("/home/jessie/internship/model/cube.urdf", xpos, ypos, zpos,
        #                                orn[0], orn[1], orn[2], orn[3])
        #     self.targetUid = p.loadURDF("/home/jessie/internship/model/cube_target.urdf",
        #                                 [xpos_target, ypos_target, zpos_target],
        #                                 orn_target, useFixedBase=1)
        p.setCollisionFilterPair(self.targetUid, self.blockUid, -1, -1, 0)
        self.goal=np.array([xpos_target,ypos_target,zpos_target])
        p.setGravity(0, 0, -10)
        self._envStepCounter = 0
        obs = self._get_obs()
        self._observation = obs
        return np.array(self._observation)

    def _set_action(self, action):
        self._barobot.applyAction(action)

    def _get_obs(self):
        self._observation = self._barobot.getObservation()
        gripperState = p.getLinkState(self._barobot.baUid, self._barobot.baFingerIndexL)
        gripperStateR = p.getLinkState(self._barobot.baUid, self._barobot.baFingerIndexR)

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

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float32)

if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env 
    env = barobotGymEnv(has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            reward_type="sparse")
    check_env(env)