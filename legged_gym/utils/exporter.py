# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import os
import torch
from torch import nn
from typing import Optional


def export_policy_as_jit(policy: object, path: str, normalizer: Optional[object] = None, filename="policy.pt"):
    """Export policy into a Torch JIT file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported JIT file. Defaults to "policy.pt".
    """
    policy_exporter = _TorchPolicyExporter(policy, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(
    policy: object, path: str, normalizer: Optional[object] = None, filename="policy.onnx", verbose=False
):
    """Export policy into a Torch ONNX file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, normalizer, verbose)
    policy_exporter.export(path, filename)



def export_policy_as_pkl(
    policy: nn.Module, path: str, filename="policy.pkl"
):
    """Export policy into a Torch pkl file.

    Args:
        policy: The policy torch module.
        normalizer: The empirical normalizer module. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported pkl file. Defaults to "policy.pkl".
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    model_dict = policy.state_dict()
    torch.save(model_dict, os.path.join(path, filename))


"""
Helper Functions
"""


def detect_algorithm(policy) -> str:
    """Detect the algorithm type from policy attributes.

    Supported algorithms:
        - PPO:          actor
        - CTS:          student_encoder + actor
        - MoE-CTS:      student_moe_encoder + actor
        - MoE-NG-CTS:   student_moe_encoder + obs_no_goal_mask + actor
        - MCP-CTS:      student_encoder + obs_no_goal_mask + actor_mcp
        - AC-MoE-CTS:   student_encoder + actor_moe
        - Dual-MoE-CTS: student_moe_encoder + actor_moe

    Detection order matters — more specific checks come first.
    """
    has_actor_mcp = hasattr(policy, "actor_mcp")
    has_actor_moe = hasattr(policy, "actor_moe")
    has_student_encoder = hasattr(policy, "student_encoder")
    has_student_moe_encoder = hasattr(policy, "student_moe_encoder")
    has_no_goal_mask = hasattr(policy, "obs_no_goal_mask")

    # MCP-CTS: actor_mcp + student_encoder + obs_no_goal_mask
    if has_actor_mcp and has_student_encoder:
        return "MCP-CTS"

    # Dual-MoE-CTS: actor_moe + student_moe_encoder
    if has_actor_moe and has_student_moe_encoder:
        return "Dual-MoE-CTS"

    # AC-MoE-CTS: actor_moe + student_encoder
    if has_actor_moe and has_student_encoder:
        return "AC-MoE-CTS"

    # MoE-NG-CTS: student_moe_encoder + obs_no_goal_mask + actor
    if has_student_moe_encoder and has_no_goal_mask:
        return "MoE-NG-CTS"

    # MoE-CTS: student_moe_encoder + actor
    if has_student_moe_encoder:
        return "MoE-CTS"

    # CTS: student_encoder + actor
    if has_student_encoder:
        return "CTS"

    # PPO: plain actor
    if hasattr(policy, "actor"):
        return "PPO"

    raise ValueError("Policy does not have a recognized actor/encoder module.")


class _TorchPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into JIT file.

    Supported algorithms:
        - PPO:          actor
        - CTS:          student_encoder + actor
        - MoE-CTS:      student_moe_encoder + actor
        - MoE-NG-CTS:   student_moe_encoder + obs_no_goal_mask + actor
        - MCP-CTS:      student_encoder + obs_no_goal_mask + actor_mcp
        - AC-MoE-CTS:   student_encoder + actor_moe
        - Dual-MoE-CTS: student_moe_encoder + actor_moe
    """

    def __init__(self, policy, normalizer=None):
        super().__init__()

        # --- Detect algorithm type ---
        algo = detect_algorithm(policy)

        # --- Copy encoder ---
        if algo in ("CTS", "MCP-CTS", "AC-MoE-CTS"):
            self.student_encoder = copy.deepcopy(policy.student_encoder).cpu()
        elif algo in ("MoE-CTS", "MoE-NG-CTS", "Dual-MoE-CTS"):
            self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder).cpu()

        # --- Copy actor ---
        if algo in ("MCP-CTS",):
            self.actor = copy.deepcopy(policy.actor_mcp).cpu()
        elif algo in ("AC-MoE-CTS", "Dual-MoE-CTS"):
            self.actor = copy.deepcopy(policy.actor_moe).cpu()
        else:
            self.actor = copy.deepcopy(policy.actor).cpu()

        # --- Copy goal mask ---
        if algo in ("MoE-NG-CTS", "MCP-CTS"):
            self.obs_no_goal_mask = copy.deepcopy(policy.obs_no_goal_mask).cpu()

        # --- Copy history buffer ---
        if algo != "PPO":
            self.history_length = policy.history.shape[1]
            self.history = torch.zeros(1, policy.history.shape[1], policy.history.shape[2], device="cpu")

        # --- Copy normalizer ---
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer).cpu()
        else:
            self.normalizer = torch.nn.Identity()

        # --- Bind forward method ---
        self.forward = {
            "PPO":          self.forward_ppo,
            "CTS":          self.forward_cts,
            "MoE-CTS":      self.forward_moe_cts,
            "MoE-NG-CTS":   self.forward_moe_no_goal_cts,
            "MCP-CTS":      self.forward_mcp_cts,
            "AC-MoE-CTS":   self.forward_ac_moe_cts,
            "Dual-MoE-CTS": self.forward_dual_moe_cts,
        }[algo]

    # --- Forward methods (one per algorithm) ---

    # PPO
    def forward_ppo(self, x):
        x = self.normalizer(x)
        return self.actor(x)

    # CTS
    def forward_cts(self, x):
        x = self.normalizer(x)
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        latent = self.student_encoder(self.history.flatten(1))
        x = torch.cat([latent, x], dim=1)
        return self.actor(x), (latent,)

    # MoE-CTS
    def forward_moe_cts(self, x):
        x = self.normalizer(x)
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        latent, weights = self.student_moe_encoder(self.history.flatten(1))
        x = torch.cat([latent, x], dim=1)
        return self.actor(x), (latent, weights)

    # MoE-NG-CTS (MoE with No-Goal mask)
    def forward_moe_no_goal_cts(self, x):
        x = self.normalizer(x)
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        history_no_goal = self.history.reshape(1, self.history_length, -1)[:, :, self.obs_no_goal_mask].reshape(1, -1)
        latent, weights = self.student_moe_encoder(self.history.flatten(1), history_no_goal)
        x = torch.cat([latent, x], dim=1)
        return self.actor(x), (latent, weights)

    # MCP-CTS
    def forward_mcp_cts(self, x):
        x = self.normalizer(x)
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        x_no_goal = x[:, self.obs_no_goal_mask]
        latent = self.student_encoder(self.history.flatten(1))
        x = torch.cat([latent, x], dim=1)
        x_no_goal = torch.cat([latent, x_no_goal], dim=1)
        mean_action, _, weights = self.actor(x, x_no_goal)
        return mean_action, (latent, weights)

    # AC-MoE-CTS
    def forward_ac_moe_cts(self, x):
        x = self.normalizer(x)
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        latent = self.student_encoder(self.history.flatten(1))
        x = torch.cat([latent, x], dim=1)
        mean, weights = self.actor(x)
        return mean, (latent, weights)

    # Dual-MoE-CTS
    def forward_dual_moe_cts(self, x):
        x = self.normalizer(x)
        self.history = torch.cat([self.history[:, 1:], x.unsqueeze(1)], dim=1)
        latent, student_weights = self.student_moe_encoder(self.history.flatten(1))
        x = torch.cat([latent, x], dim=1)
        mean, actor_weights = self.actor(x)
        return mean, (latent, student_weights, actor_weights)

    # --- Reset & Export ---

    @torch.jit.export
    def reset(self):
        if hasattr(self, 'history'):
            self.history = torch.zeros_like(self.history)

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class _OnnxPolicyExporter(torch.nn.Module):
    """Exporter of actor-critic into ONNX file.

    Supported algorithms:
        - PPO:          actor
        - CTS:          student_encoder + actor
        - MoE-CTS:      student_moe_encoder + actor
        - MoE-NG-CTS:   student_moe_encoder + obs_no_goal_mask + actor
        - MCP-CTS:      student_encoder + obs_no_goal_mask + actor_mcp
        - AC-MoE-CTS:   student_encoder + actor_moe
        - Dual-MoE-CTS: student_moe_encoder + actor_moe
    """

    def __init__(self, policy, normalizer=None, verbose=False):
        super().__init__()
        self.verbose = verbose
        self.normalizer = torch.nn.Identity()

        # --- Detect algorithm type and copy components ---
        algo = detect_algorithm(policy)

        # Copy encoder
        if algo in ("CTS", "MCP-CTS", "AC-MoE-CTS"):
            self.student_encoder = copy.deepcopy(policy.student_encoder)
        elif algo in ("MoE-CTS", "MoE-NG-CTS", "Dual-MoE-CTS"):
            self.student_moe_encoder = copy.deepcopy(policy.student_moe_encoder)

        # Copy actor
        if algo in ("MCP-CTS",):
            self.actor = copy.deepcopy(policy.actor_mcp)
        elif algo in ("AC-MoE-CTS", "Dual-MoE-CTS"):
            self.actor = copy.deepcopy(policy.actor_moe)
        else:
            self.actor = copy.deepcopy(policy.actor)

        # Copy goal mask
        if algo in ("MoE-NG-CTS", "MCP-CTS"):
            self.obs_no_goal_mask = copy.deepcopy(policy.obs_no_goal_mask).cpu()

        # Copy history shape info
        if algo != "PPO":
            self.history_length = policy.history.shape[1]
            self.obs_dim = policy.history.shape[2]
            self.input_dim = self.history_length * self.obs_dim
        else:
            self.input_dim = self.actor[0].in_features

        # Bind forward method
        self._algo = algo
        self.forward = {
            "PPO":          self.forward_ppo,
            "CTS":          self.forward_cts,
            "MoE-CTS":      self.forward_moe_cts,
            "MoE-NG-CTS":   self.forward_moe_no_goal_cts,
            "MCP-CTS":      self.forward_mcp_cts,
            "AC-MoE-CTS":   self.forward_ac_moe_cts,
            "Dual-MoE-CTS": self.forward_dual_moe_cts,
        }[algo]

    # --- Forward methods (one per algorithm) ---

    # PPO
    def forward_ppo(self, x):
        x = self.normalizer(x)
        return self.actor(x)

    # CTS
    def forward_cts(self, x):
        x = self.normalizer(x)
        last_obs = x[:, -self.obs_dim:]
        latent = self.student_encoder(x)
        x = torch.cat([latent, last_obs], dim=1)
        return self.actor(x), latent

    # MoE-CTS
    def forward_moe_cts(self, x):
        x = self.normalizer(x)
        last_obs = x[:, -self.obs_dim:]
        latent, weights = self.student_moe_encoder(x)
        x = torch.cat([latent, last_obs], dim=1)
        return self.actor(x), latent, weights

    # MoE-NG-CTS (MoE with No-Goal mask)
    def forward_moe_no_goal_cts(self, x):
        x = self.normalizer(x)
        last_obs = x[:, -self.obs_dim:]
        history_3d = x.view(-1, self.history_length, self.obs_dim)
        history_no_goal = history_3d[:, :, self.obs_no_goal_mask].reshape(x.shape[0], -1)
        latent, weights = self.student_moe_encoder(x, history_no_goal)
        x = torch.cat([latent, last_obs], dim=1)
        return self.actor(x), latent, weights

    # MCP-CTS
    def forward_mcp_cts(self, x):
        x = self.normalizer(x)
        last_obs = x[:, -self.obs_dim:]
        obs_no_goal = last_obs[:, self.obs_no_goal_mask]
        latent = self.student_encoder(x)
        x_in = torch.cat([latent, last_obs], dim=1)
        x_no_goal_in = torch.cat([latent, obs_no_goal], dim=1)
        mean_action, _, weights = self.actor(x_in, x_no_goal_in)
        return mean_action, latent, weights

    # AC-MoE-CTS
    def forward_ac_moe_cts(self, x):
        x = self.normalizer(x)
        last_obs = x[:, -self.obs_dim:]
        latent = self.student_encoder(x)
        x = torch.cat([latent, last_obs], dim=1)
        mean, weights = self.actor(x)
        return mean, latent, weights

    # Dual-MoE-CTS
    def forward_dual_moe_cts(self, x):
        x = self.normalizer(x)
        last_obs = x[:, -self.obs_dim:]
        latent, student_weights = self.student_moe_encoder(x)
        x = torch.cat([latent, last_obs], dim=1)
        mean, actor_weights = self.actor(x)
        return mean, latent, student_weights, actor_weights

    # --- ONNX export ---

    # Output names per algorithm
    _OUTPUT_NAMES = {
        "PPO":          ["actions"],
        "CTS":          ["actions", "latent"],
        "MoE-CTS":      ["actions", "latent", "weights"],
        "MoE-NG-CTS":   ["actions", "latent", "weights"],
        "MCP-CTS":      ["actions", "latent", "weights"],
        "AC-MoE-CTS":   ["actions", "latent", "weights"],
        "Dual-MoE-CTS": ["actions", "latent", "student_weights", "actor_weights"],
    }

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.input_dim)
        output_names = self._OUTPUT_NAMES[self._algo]

        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=output_names,
            dynamic_axes={},
        )
