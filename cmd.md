## Train
```bash
# MoE CTS
python legged_gym/scripts/train.py --task g1_moe_cts --num_envs 8096 --headless --robogauge
```
## Play
```bash
# MoE CTS
python legged_gym/scripts/play.py --task g1_moe_cts --num_envs 8 --load_run Nov13_00-00-05_  # load specified run
``