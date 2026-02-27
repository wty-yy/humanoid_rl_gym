from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO, LeggedRobotCfgCTS, LeggedRobotCfgMoENGCTS, LeggedRobotCfgMoENGCTS, LeggedRobotCfgMCPCTS, LeggedRobotCfgACMoECTS, LeggedRobotCfgDualMoECTS, LeggedRobotCfgMoECTS

ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

class G1Cfg(LeggedRobotCfg):
    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.793] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            # lower body
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            # upper body
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.25,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.97,
            "left_wrist_roll_joint": 0.15,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.25,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.97,
            "right_wrist_roll_joint": -0.15,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }

    class env(LeggedRobotCfg.env):
        num_envs = 8192
        # ang_vel(3) + gravity(3) + commands(3) + dof_pos(29) + dof_vel(29) + previous_actions(29)
        num_observations = 3 + 3 + 3 + 29 + 29 + 29 # 96
        # base_lin_vel(3) + obs(96) + foot_contact_forces(2) + torques(29) + motor_accelerations(29) + height_measurements(187)
        num_privileged_obs = 3 + 96 + 2 + 29 + 29 + 187  # 346

        num_actions = 29
        episode_length_s = 25

    class domain_rand(LeggedRobotCfg.domain_rand):
        ### Robot properties ###
        randomize_friction = True
        friction_range = [0.0, 2.0]

        randomize_base_mass = True
        added_mass_range = [-1., 1.]

        randomize_link_mass = True
        multiplied_link_mass_range = [0.9, 1.1]

        randomize_base_com = True
        added_base_com_range = [-0.03, 0.03]

        randomize_restitution = True # restitution to robot links (Robot init)
        restitution_range = [0.0, 0.5]

        ### Environment reset ###
        randomize_pd_gains = True
        stiffness_multiplier_range = [0.9, 1.1]  
        damping_multiplier_range = [0.9, 1.1]    

        randomize_motor_zero_offset = True
        motor_zero_offset_range = [-0.035, 0.035]

        randomize_motor_strength = True # (Env reset)
        motor_strength_range = [0.8, 1.2]

        ### Environment step ###
        push_robots = True
        push_interval_s = 4
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6

        randomize_action_delay = True # use last_action with 0~20 ms delay, 4 decimation

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'

        NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 62.8318530718
        DAMPING_RATIO = 2.0

        STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2         # 14.2506
        STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2   # 40.1792
        STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2   # 99.0984
        STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2         # 16.7783

        DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ       # 0.9072
        DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ # 2.5579
        DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ # 6.3088
        DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ       # 1.0681

        stiffness = {  # [N*m/rad]
            'hip_yaw': STIFFNESS_7520_14,         # 40.1792
            'hip_roll': STIFFNESS_7520_22,        # 99.0984
            'hip_pitch': STIFFNESS_7520_14,       # 40.1792
            'knee': STIFFNESS_7520_22,            # 99.0984
            'ankle': 2.0 * STIFFNESS_5020,        # 28.5012
            'waist': 2.0 * STIFFNESS_5020,        # 28.5012 (waist_roll, waist_pitch)
            'waist_yaw': STIFFNESS_7520_14,       # 40.1792
            'shoulder': 50,           # 14.2506
            'elbow': STIFFNESS_5020,              # 14.2506
            'wrist': STIFFNESS_4010,              # 16.7783 (wrist_pitch, wrist_yaw)
            'wrist_roll': STIFFNESS_5020,         # 14.2506
        }
        damping = {  # [N*m*s/rad]
            'hip_yaw': DAMPING_7520_14,           # 2.5579
            'hip_roll': DAMPING_7520_22,          # 6.3088
            'hip_pitch': DAMPING_7520_14,         # 2.5579
            'knee': DAMPING_7520_22,              # 6.3088
            'ankle': 2.0 * DAMPING_5020,          # 1.8144
            'waist': 2.0 * DAMPING_5020,          # 1.8144 (waist_roll, waist_pitch)
            'waist_yaw': DAMPING_7520_14,         # 2.5579
            'shoulder': DAMPING_5020,             # 0.9072
            'elbow': DAMPING_5020,                # 0.9072
            'wrist': DAMPING_4010,                # 1.0681 (wrist_pitch, wrist_yaw)
            'wrist_roll': DAMPING_5020,           # 0.9072
        }
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
    
    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane' # none, plane, heightfield or trimesh
        
    class commands(LeggedRobotCfg.commands):
        num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5. # time before command are changed[s]
        heading_command = False # if true: compute ang vel command from heading error
        # start training with zero commands and then gradually increase zero command probability
        zero_command_curriculum = {'start_iter': 0, 'end_iter': 1500, 'start_value': 0.0, 'end_value': 0.1}
        limit_ang_vel_at_zero_command_prob = 0.2 # probability of add limiting angular velocity commands when zero command is sampled
        limit_vel_prob = 0.2 # probability of limiting linear velocity command
        limit_vel_invert_when_continuous = True # invert the limit logic when using continuous sample limit velocity commands
        limit_vel = {"lin_vel_x": [-1, 1], "lin_vel_y": [-1, 1], "ang_vel_yaw": [-1, 0, 1]} # sample vel commands from min [-1] or zero [0] or max [1] range only
        stop_heading_at_limit = True # stop heading updates when vel is limited
        dynamic_resample_commands = True # sample commands with low bounds
        command_range_curriculum = [{ # list for command range curriculums at specific training iterations
            'iter': 5000, # training iteration at which the command ranges are updated
            'lin_vel_x': [-0.5, 0.5], # min max [m/s]
            'lin_vel_y': [-0.3, 0.3], # min max [m/s]
            'ang_vel_yaw': [-1.0, 1.0], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        }, { # list for command range curriculums at specific training iterations
            'iter': 15000, # training iteration at which the command ranges are updated
            'lin_vel_x': [-1.2, 1.2], # min max [m/s]
            'lin_vel_y': [-0.5, 0.5], # min max [m/s]
            'ang_vel_yaw': [-1.5, 1.5], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        }, {
            'iter': 25000, # training iteration at which the command ranges are updated
            'lin_vel_x': [-2.0, 2.0], # min max [m/s]
            'lin_vel_y': [-0.8, 0.8], # min max [m/s]
            'ang_vel_yaw': [-2.0, 2.0], # min max [rad/s]
            'heading': [-1.57, 1.57], # min max [rad]
        }]
        # [wave, slope, rough slope, stairs up, stairs down, obstacles, stepping stones, gap, flat]
        gait_phase = 0.64  # cycle time for gait phase [s]
        terrain_max_command_ranges = [
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.5, 1.5], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # wave
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.5, 1.5], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # slope
            {'lin_vel_x': [-1.5, 1.5], 'lin_vel_y': [-1.5, 1.5], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # rough slope
            {'lin_vel_x': [-0.8, 0.8], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs up
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stairs down
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # obstacles
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # stepping stones
            {'lin_vel_x': [-1.0, 1.0], 'lin_vel_y': [-1.0, 1.0], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # gap
            {'lin_vel_x': [-2.0, 2.0], 'lin_vel_y': [-1.5, 1.5], 'ang_vel_yaw': [-1.5, 1.5], 'heading': [-1.57, 1.57]},  # flat
        ]

        class ranges:
            lin_vel_x = [-0.3, 0.3] # min max [m/s]
            lin_vel_y = [-0.1, 0.1] # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]   # min max [rad/s]
            heading = [-1.57, 1.57] # min max [rad]
        
    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/g1/g1_29dof_mode_15.urdf'
        name = "g1"
        foot_name = "ankle_roll"
        penalize_contacts_on = ["hip", "knee"]
        terminate_after_contacts_on = []
        terminate_base_height = 0.3
        lower_body_joint_names = ["hip", "knee", "ankle"]
        upper_body_joint_names = ["waist", "shoulder", "elbow", "wrist"]

        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        armatures_overwrite = {
            "hip_yaw": ARMATURE_7520_14,
            "hip_roll": ARMATURE_7520_22,
            "hip_pitch": ARMATURE_7520_14,
            "knee": ARMATURE_7520_22,
            "ankle": 2.0 * ARMATURE_5020, # ankle_pitch, ankle_roll
            "waist": 2.0 * ARMATURE_5020, # waist_roll, waist_pitch
            "waist_yaw": ARMATURE_7520_14,
            "shoulder": ARMATURE_5020, # shoulder_pitch, shoulder_roll, shoulder_yaw
            "elbow": ARMATURE_5020,
            "wrist_roll": ARMATURE_5020,
            "wrist_pitch": ARMATURE_4010,
            "wrist_yaw": ARMATURE_4010,
        }
  
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.78
        only_positive_rewards = False
        max_contact_force = 343. # forces above this value are penalized, g1 weight 35kg
        curriculum_rewards = [
            {'reward_name': 'lin_vel_z', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
            {'reward_name': 'correct_base_height', 'start_iter': 0, 'end_iter': 5000, 'start_value': 1.0, 'end_value': 10.0},
            # {'reward_name': 'dof_power', 'start_iter': 0, 'end_iter': 3000, 'start_value': 1.0, 'end_value': 0.1},
            # {'reward_name': 'upright', 'start_iter': 0, 'end_iter': 1500, 'start_value': 1.0, 'end_value': 0.0},
        ]
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        dynamic_sigma = None
        upper_body_to_default = {
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.20,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 1.36,
            "left_wrist_roll_joint": 0.15,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.20,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 1.36,
            "right_wrist_roll_joint": -0.15,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
        }
        upper_body_scaler = {
            "waist_yaw_joint": 3.0,
            "waist_roll_joint": 3.0,
            "waist_pitch_joint": 3.0,
            "left_shoulder_pitch_joint": 1.0,
            "left_shoulder_roll_joint": 1.0,
            "left_shoulder_yaw_joint": 1.0,
            "left_elbow_joint": 1.0,
            "left_wrist_roll_joint": 1.0,
            "left_wrist_pitch_joint": 1.0,
            "left_wrist_yaw_joint": 1.0,
            "right_shoulder_pitch_joint": 1.0,
            "right_shoulder_roll_joint": 1.0,
            "right_shoulder_yaw_joint": 1.0,
            "right_elbow_joint": 1.0,
            "right_wrist_roll_joint": 1.0,
            "right_wrist_pitch_joint": 1.0,
            "right_wrist_yaw_joint": 1.0,
        }
        stance_body_to_default = {
            "left_hip_pitch_joint": -0.1,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.3,
            "left_ankle_pitch_joint": -0.2,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.1,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.3,
            "right_ankle_pitch_joint": -0.2,
            "right_ankle_roll_joint": 0.0,
            **upper_body_to_default,
        }
        class scales:
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            dof_acc = -2.5e-7
            dof_power = -2e-5
            torques = -1e-4
            correct_base_height = -1.0
            action_rate = -0.01
            action_smoothness = -0.01
            collision = -1.0
            dof_pos_limits = -2.0
            feet_regulation = -0.05

            upper_body_to_default = -0.1
            stance_to_default = -0.2
            parallel_feet = -0.2
            orientation_xy = -2.0
            feet_diff_height = -0.05

    class noise(LeggedRobotCfg.noise):
        add_noise = True

class G1TerrainCfg(G1Cfg):
    class terrain(LeggedRobotCfg.terrain):
        max_init_terrain_level = 5
        # [wave, slope, rough_slope, stairs up, stairs down, obstacles, stepping_stones, gap, flat]
        terrain_proportions = [0.05, 0.20, 0.05, 0.25, 0.10, 0.20, 0.0, 0.0, 0.15]  # 这个更偏向平地斜坡
        # terrain_proportions = [0.15, 0.20, 0.10, 0.0, 0.20, 0.20, 0.0, 0.0, 0.15]  # 去除上楼梯
        move_down_by_accumulated_xy_command = True # move down the terrain curriculum based on accumulated xy command distance instead of absolute distance

class G1CfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'g1_ppo'
        max_iterations = 50000
        save_interval = 500

class G1CfgCTS(LeggedRobotCfgCTS):
    class runner(LeggedRobotCfgCTS.runner):
        num_steps_per_env = 24
        run_name = ''
        experiment_name = 'g1_cts'
        max_iterations = 50000
        save_interval = 500
    
    class policy(LeggedRobotCfgCTS.policy):
        latent_dim = 32
        norm_type = 'l2norm'

class G1CfgMoECTS(LeggedRobotCfgMoECTS):
    class policy(LeggedRobotCfgMoECTS.policy):
        expert_num = 8  # number of experts in the student model
    
    class runner(LeggedRobotCfgMoECTS.runner):
        run_name = ''
        experiment_name = 'g1_moe_cts'
        max_iterations = 50000
        save_interval = 500
