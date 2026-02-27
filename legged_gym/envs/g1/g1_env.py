
from legged_gym.envs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import torch

class G1Robot(LeggedRobot):
    def _get_phase(self):
        cycle_time = self.cfg.commands.gait_phase
        phase = self.episode_length_buf * self.dt / cycle_time
        return phase

    def _get_noise_scale_vec(self, cfg):
        noise_vec = torch.zeros_like(self.obs_buf[0])
        self.add_noise = self.cfg.noise.add_noise
        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec[:3] = noise_scales.ang_vel * noise_level * self.obs_scales.ang_vel
        noise_vec[3:6] = noise_scales.gravity * noise_level
        noise_vec[6:9] = 0. # commands
        noise_vec[9:9+self.num_actions] = noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos
        noise_vec[9+self.num_actions:9+2*self.num_actions] = noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel
        noise_vec[9+2*self.num_actions:9+3*self.num_actions] = 0. # previous actions
        noise_vec[9+3*self.num_actions:9+3*self.num_actions+2] = 0. # gait phase

        return noise_vec
    
    def compute_observations(self):
        """ Computes observations
        """
        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
        ), dim=-1)
        
        heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0) * self.obs_scales.height_measurements
        
        self.privileged_obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel  * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) * 1e-3,  # foot contact forces (4,)
            self.torques / self.torque_limits,  # motor torques (12,)
            (self.last_dof_vel - self.dof_vel) / self.dt * 1e-4,  # motor accelerations (12,)
            heights,  # height measurements (187,)
        ), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_upper_body_to_default(self):
        upper_body_pos = self.dof_pos[:, self.upper_body_dof_indices]
        default_upper_body_pos = self.upper_body_rew_pos.unsqueeze(0)
        upper_body_scaler = self.upper_body_scaler.unsqueeze(0)
        return torch.sum(torch.abs(upper_body_pos - default_upper_body_pos) * upper_body_scaler, dim=1)

    def _reward_stance_to_default(self):
        stance = (self.commands[:, :3] == 0.0).all(dim=1)
        default_stance_pos = self.stance_body_rew_pos.unsqueeze(0)
        return torch.sum(torch.abs(self.dof_pos - default_stance_pos), dim=1) * stance

    def _reward_parallel_feet(self):
        # feet_indices = [6, 12]
        rigid_body_states = self.rigid_body_states.view(self.num_envs, self.num_bodies, 13)
        left_foot_quat = rigid_body_states[:, self.feet_indices[0], 3:7]
        right_foot_quat = rigid_body_states[:, self.feet_indices[1], 3:7]

        # left x-axis -> world frame -> right foot frame
        left_forward_world = quat_apply(left_foot_quat, self.forward_vec)
        left_forward_right_frame = quat_rotate_inverse(right_foot_quat, left_forward_world)

        # left x-axis in right foot frame's x-y plane angle
        theta = torch.atan2(left_forward_right_frame[:, 1], left_forward_right_frame[:, 0])
        # is_pigeon_toed = theta < 0

        threshold = 0.1  # [rad]
        rew = theta.abs() * (theta.abs() > threshold)
        cmd_ang_err = torch.exp(-torch.square(self.commands[:, 2]) * 10)
        rew = rew * cmd_ang_err
        return rew