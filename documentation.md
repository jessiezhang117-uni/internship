class MJ_controller:
	groups:
		arm
		gripper
	actuated_joint_ids
	current_output
	actutaors:
		0 acuator ID
		1 actuator name
		2 joint ID of the joint controller by this actuator
		3 joint name
		4 controller for controlling the actuator
- create_group(): allows the user to create custom objects for controlling groups of joints
- show_model_info(): display relevant model info for the user, namely bodies, joints, actuators, as well as their IDs and ranges
- create_lists(): creates some basic lists and fill them with initial values. The following lists/dictionaries are created: 
	- controller_list: contains a controller for each of the actuated joints.
	- current_target_joint_values: same as the current setpoints for all controllers, created for convenience.
	- current_output: a list containing the output values of all the controllers.
	- actuators:
		0 actuator ID
		1 actuator name
		2 joint ID of the joint controlled by this actuator
		3 joint name
		4 controller for controlling the actuator

- actuate_joint_group(self,group,motor_values): 

- move_group_to_joint_target(self,group,target,tolerance,max_steps,plot,render,quiet): moves the specified joint group to a joint target
	- groups: string specifying the group to move
	- target: list of target joint values for the group
	- tolerance: threshold within which the error of each joint must be before the method finishes
	- max_steps: maximum number of steps to actuate before breaking 
	- plot: if True, a .png image of the group joint trajectories will be saved to the local directory (This can be used for PID tuning in case of overshooting)
	- 
	current_joint_values = sim.data.qpos[actuated_joint_ids]

- set_group_joint_target(self,group,target)

- open_gripper(self,half): opens the gripper while keeping the arm in a steady position
-
 close_gripper(self): closes the gripper while keeping the arm in a steady position

- grasp(self): attempts a grasp at the current location and prints some feedback on wheather it was successful

- move_ee(ee_position):  moves the robot arm so that the gripper center ends up at the requested XYZ-position with a vertical gripper position. #ee_position: list of XYZ-coordinates of the end-effector
	joint_angles

- ik(ee_position): method for solving simple inverse kinematic problems (top down grasping)
	current_carthesian_target
	ee_position_base
	gripper_center_position
	joint_angles: list of joint angles that will achieve the desired ee-position

- ik2(self,pose_target):

- stay(self,duration): holds the current position by actuating the joints towards their current target position

- fill_plot_list(self,group,step): creates a two dimensional list of joint angles for plotting

- create_joint_angle_plot(): saves the recorded joint values as a .png file


class GraspEnv:
- step(self,action): lets the agent execute the action
	action: the action to be performed
	observation: np-array containing current joint position and orientation
	rewards: the reward obtained
	done: fla indicating wheather the episode has finished or not
- set_action_space():
- set_grasp_position(self,position): not used for later
- rotate_wrist_3_joint_to_value(sele,degree):
- move_and_grasp(coordinates,rotation):
- get_observation(self): 
- reset_model(self): methods to perform additional reset steps and return an observation

	
	
MJ_Controller: This class can be used as a standalone class for basic robot control in MuJoCo. This can be useful for trying out models and their grasping capabilities. Alternatively, its methods can also be used by any other class (like a Gym environment) to provide some more functionality. One example of this might be to move the robot back into a certain position after every episode of training, which might be preferable compared to just resetting all the joint angles and velocities. 

GraspEnv: A Gym environment for training reinforcement learning agents. The task to master is a pick & place object.  


To Do:
1. fix the inverse kinematic function:
	joints and links conflict 
	position and orientation
2. package compatible issue between gym and mujoco
3. add DQN & DDPG with stable_baselines