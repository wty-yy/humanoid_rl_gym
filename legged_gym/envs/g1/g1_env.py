
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
        phase = self._get_phase()
        gait_phase = torch.stack((torch.sin(2 * np.pi * phase), torch.cos(2 * np.pi * phase)), dim=-1)
        self.obs_buf = torch.cat((
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, :3] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.actions,
            gait_phase,
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
            gait_phase,
            torch.norm(self.contact_forces[:, self.feet_indices, :], dim=-1) * 1e-3,  # foot contact forces (4,)
            self.torques / self.torque_limits,  # motor torques (12,)
            (self.last_dof_vel - self.dof_vel) / self.dt * 1e-4,  # motor accelerations (12,)
            heights,  # height measurements (187,)
        ), dim=-1)

        if self.add_noise:
            self.obs_buf += (2 * torch.rand_like(self.obs_buf) - 1) * self.noise_scale_vec

    def _reward_upper_body_to_default(self):
        upper_body_pos = self.dof_pos[:, self.upper_body_dof_indices]
        default_upper_body_pos = self.default_dof_pos[:, self.upper_body_dof_indices]
        return torch.sum(torch.abs(upper_body_pos - default_upper_body_pos), dim=1)
