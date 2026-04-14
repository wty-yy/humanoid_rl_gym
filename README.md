<div align="center">
	<h1 align="center">Humanoid RL GYM</h1>
	<p align="center">
		<span>🌎 English</span> | <a href="README_zh.md">🇨🇳 中文</a>
	</p>
</div>

<p align="center">
	<strong>This repository builds on <a href="https://github.com/wty-yy/go2_rl_gym">go2_rl_gym</a> to train the Unitree G1 humanoid with reinforcement learning.</strong>
</p>

<div align="center">

</div>

## 📦 Installation

Follow the step-by-step setup guide in [setup.md](doc/setup_en.md).

## 🛠️ Usage Guide

### 1. Train

Run the following command to launch training:

```bash
python legged_gym/scripts/train.py --task g1_moe_cts
```

#### ⚙️  Arguments
- `--task`: Required. Options include `g1_moe_cts` is the paper's final version.
- `--headless`: Render viewer by default; set to `true` to disable rendering for higher throughput.
- `--resume`: Resume training from a chosen checkpoint in the logs.
- `--experiment_name`: Experiment folder to save/load from.
- `--run_name`: Run subfolder name to save/load from.
- `--load_run`: Name of the run to load (defaults to the most recent run).
- `--checkpoint`: Checkpoint index to load (defaults to the latest file).
- `--num_envs`: Number of parallel simulated environments.
- `--seed`: Random seed.
- `--max_iterations`: Maximum training iterations.
- `--sim_device`: Physics simulation device. Use `--sim_device cpu` to force CPU.
- `--rl_device`: RL computation device. Use `--rl_device cpu` to force CPU.

**Default checkpoint path**: `logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`

---

### 2. Play

Visualize policies inside Gym with:

```bash
python legged_gym/scripts/play.py --task g1_moe_cts
```

**Notes**

- Play launches on randomized terrain with difficulty between 7 and 9.
- It automatically loads the latest checkpoint inside the experiment folder.
- Override via `experiment_name` and `checkpoint`, for example:
	```bash
	python legged_gym/scripts/play.py --task g1_moe_cts --num_envs 100 --experiment_name g1_moe_cts --checkpoint 100000
	```

#### 💾 Policy Export

Play exports the Actor network to `logs/{experiment_name}/exported/policies`:
- `policy.pt`: TorchScript model for Sim2Sim.
- `policy.onnx`: ONNX model for Sim2Real.
- `policy.pkl`: Raw weights.

---

### 3. Sim2Sim & Sim2Real

#### 3.1 C++ Deployment

Follow the usage described in [unitree_cpp_deploy/deploy/g1](https://github.com/wty-yy/unitree_cpp_deploy/tree/main/deploy/robots/g1).

#### Demonstration



---

## 🎉  Acknowledgements

This repository would not exist without the following open-source projects:

- [unitree_rl_gym](https://github.com/unitreerobotics/unitree_rl_gym): Unitree's core RL training framework.
- [legged_gym](https://github.com/leggedrobotics/legged_gym): Base locomotion environment.
- [rsl_rl](https://github.com/leggedrobotics/rsl_rl.git): Reinforcement learning algorithms.
- [mujoco](https://github.com/google-deepmind/mujoco.git): High-performance CPU physics simulator.
- [unitree_sdk2_python](https://github.com/unitreerobotics/unitree_sdk2_python.git): Python hardware interface for deployment.
- [unitree_sdk2](https://github.com/unitreerobotics/unitree_sdk2): C++ hardware interface for deployment.

Related publications implemented in this repo:
- [CTS: Concurrent Teacher-Student Reinforcement Learning for Legged Locomotion](https://arxiv.org/pdf/2405.10830)

Contributors:
- [@windigal](https://github.com/windigal): CTS algorithm reproduction, terrain generation, video editing
- [@wertyuilife2](https://github.com/wertyuilife2): CTS algorithm reproduction

---

## 📄  Citation
If you find our work helpful, please cite:
```bibtex
@article{wu2026robogauge,
      title={Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion}, 
      author={Tianyang Wu and Hanwei Guo and Yuhang Wang and Junshu Yang and Xinyang Sui and Jiayi Xie and Xingyu Chen and Zeyang Liu and Xuguang Lan},
      year={2026},
      journal={arXiv preprint arXiv:2602.00678},
      url={https://arxiv.org/abs/2602.00678}, 
}
```

## 🔖  License

New contributions follow the [MIT License](LICENSE); the original unitree_rl_gym remains under the [BSD 3-Clause License](LICENSE).

See the complete [LICENSE file](LICENSE) for details.

