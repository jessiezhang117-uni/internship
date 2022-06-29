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


largeValObservation = 100

RENDER_HEIGHT = 720
RENDER_WIDTH = 960

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class barobotGymEnv(gym.Env):
    def __init__(
        self, 
        n_substeps, gripper_extra_height, block_gripper,
        has_object, target_in_the_air, target_offset, obj_range, target_range,
        distance_threshold,reward_type,renders=False,isDiscrete=False
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
        self._renders = renders
        self._isDiscrete = isDiscrete
        self._cam_dist = 2
        self._cam_yaw = 180
        self._cam_pitch = -40
        
        self._p = p
        if self._renders:
            cid = p.connect(p.SHARED_MEMORY)
            if (cid < 0):
                cid = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(2, 180, -41, [0.52, -0.2, -0.33])
        else:
            p.connect(p.DIRECT)

        self.seed()
        # observationDim = len(self._get_obs())
        # observation_high = np.array([largeValObservation] * observationDim)
        #加载机器人模型
        self._barobot = braccio_arm.braccio_arm_v0()
        self._timeStep= 1. / 240.
        
        # action_high = np.array([self._action_bound] * action_dim)
        # self.action_space = spaces.Box(-action_high, action_high)
        if (self._isDiscrete):
            self.action_space = spaces.Discrete(7)
        else:
            action_dim = 4
            self._action_bound = 0.5
            self.action_space = spaces.Box(-1.0, 1.0, shape=(action_dim,), dtype=np.float64)
        # self.observation_space = spaces.Box(-observation_high, observation_high)
        #重置环境
        # self.reset()
        obs = self.reset()  # required for init; seed can be changed later
        observation_shape = obs["observation"].shape
        # self.observation_space = spaces.Box(-10.0, 10.0, shape=observation_shape, dtype=np.float32)
        achieved_goal_shape = obs["achieved_goal"].shape
        desired_goal_shape = obs["achieved_goal"].shape
        self.observation_space = gym.spaces.Dict(
            dict(
                observation=gym.spaces.Box(-np.inf, np.inf, shape=observation_shape,dtype=np.float64),
                desired_goal=gym.spaces.Box(-np.inf, np.inf, shape=achieved_goal_shape,dtype=np.float64),
                achieved_goal=gym.spaces.Box(-np.inf, np.inf, shape=desired_goal_shape,dtype=np.float64),
            )
        )
        # GoalEnv methods
    # ----------------------------

    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float64)
        else:
            return -d

    def step(self, action):
        action = np.clip(action,-0.5,0.5)
        if p.getClosestPoints(self._barobot.baUid,self.blockUid,0.0001):
            action[3]=-1
        self._set_action(action)
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
        return self._observation

    def _set_action(self, action):
        self._barobot.applyAction(action)

    def _get_obs(self):
        # 关于机械臂的状态观察，可以从以下几个维度进行考虑
        # 末端位置、夹持器状态位置、物体位置、物体姿态、  物体相对末端位置、物体线速度、物体角速度、末端速度、物体相对末端线速度
        # 末端位置 3vec 及速度
        end_pos = np.array(self._barobot.getObservation())
        end_pos = end_pos[:3]
        # 夹持器位置、姿姿 vec3 *2  ，可能需要重新考虑一下末端位置和grip的关系
        gripperState = p.getLinkState(self._barobot.baUid, self._barobot.baGripperIndex,
                                      computeLinkVelocity=1)
        gripperPos = np.array(gripperState[4])
        gripperOrn_temp = np.array(gripperState[5])
        gripper_linear_Velocity = np.array(gripperState[6])
        gripper_angular_Velocity = np.array(gripperState[7])
        gripperOrn = p.getEulerFromQuaternion(gripperOrn_temp)
        gripperOrn = np.array(gripperOrn)
        # 物体位置、姿态
        blockPos, blockOrn_temp = p.getBasePositionAndOrientation(self.blockUid)
        blockPos = np.array(blockPos)
        blockOrn = p.getEulerFromQuaternion(blockOrn_temp)
        blockOrn = np.array(blockOrn)
        # 物体相对位置 vec *3
        relative_pos = blockPos - gripperPos
        #relative_orn = blockOrn - gripperOrn
        # block linear velocity and angular velocity 
        block_Velocity = p.getBaseVelocity(self.blockUid)
        block_linear_velocity = np.array(block_Velocity[0])
        target_pos = np.array(p.getBasePositionAndOrientation(self.targetUid)[0])
        block_angular_velocity = np.array(block_Velocity[1])

        # blockEulerInGripper = p.getEulerFromQuaternion(blockOrnInGripper)
        # #we return the relative x,y position and euler angle of block in gripper space
        # blockInGripperPosXYEulZ = [blockPosInGripper[0], blockPosInGripper[1], blockEulerInGripper[2]]
        # self._observation.extend(list(blockInGripperPosXYEulZ))
        # to numpy.array()
        #
        obs = [
            end_pos.flatten(),
            #gripperPos.flatten(),
            gripperOrn.flatten(),
            gripper_linear_Velocity.flatten(),
            gripper_angular_Velocity.flatten(),
            blockPos.flatten(),
            blockOrn.flatten(),
            relative_pos.flatten(),
            #relative_orn.flatten(),
            #target_pos.flatten(),
            #target_relative_pos.flatten()
            block_linear_velocity.flatten(),
            block_angular_velocity.flatten(),
            # block_relative_linear_velocity.flatten()
        ]
        if not self.has_object:
            achieved_goal = end_pos.copy()
        else:
            achieved_goal = blockPos.copy()
        for i in range(1, len(obs)):
            end_pos = np.append(end_pos, obs[i])
        obs = end_pos.reshape(-1)
        self._observation = obs
        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': target_pos.flatten(),
        }

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def _is_success(self, achieved_goal, desired_goal):
        d = goal_distance(achieved_goal, desired_goal)
        return (d < self.distance_threshold).astype(np.float64)


    def render(self, mode="rgb_array", close=False):
        if mode != "rgb_array":
            return np.array([])

        base_pos, orn = self._p.getBasePositionAndOrientation(self._barobot.baUid)
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

if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env 
    env = barobotGymEnv(has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.15, distance_threshold=0.05,
            reward_type="sparse")
    check_env(env)
