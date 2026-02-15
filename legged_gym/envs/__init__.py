from legged_gym.utils.task_registry import task_registry

from legged_gym.envs.g1.g1_env import G1Robot
from legged_gym.envs.g1.g1_config import G1Cfg, G1CfgPPO, G1CfgCTS, G1CfgMoECTS

task_registry.register("g1", G1Robot, G1Cfg(), G1CfgPPO())
task_registry.register("g1_cts", G1Robot, G1Cfg(), G1CfgCTS())
task_registry.register("g1_moe_cts", G1Robot, G1Cfg(), G1CfgMoECTS())
# task_registry.register("g1_moe_ng_cts", G1Robot, G1Cfg(), G1CfgMoENGCTS())
# task_registry.register("g1_mcp_cts", G1Robot, G1Cfg(), G1CfgMCPCTS())
# task_registry.register("g1_ac_moe_cts", G1Robot, G1Cfg(), G1CfgACMoECTS())
# task_registry.register("g1_dual_moe_cts", G1Robot, G1Cfg(), G1CfgDualMoECTS())
