# 20260217
## v0.0.3.2
1. 训练最大速度达到2.0m/s
# 20260216
## v0.0.3.1
1. 更新upper body奖励关节位置，避免手臂向前过度
2. 加入sim2sim的g1测试
## v0.0.3
1. 参考unitree_rl_lab发现电机的kp和armature存在问题，加入对armature配置的功能`armatures_overwrite`，对齐unitree_rl_lab中的mimic电机精确配置方案，通过测试发现，g1的damping非常大超过了1.0，而go2的damping为0.5无需armature，而g1必须要armature大约在0.01，而由于不同位置使用了不同型号电机，因此进一步加入精确配置
# 20260215
## v0.0.2
1. 打开自碰撞后就不能根据接触力判断环境终止，修改为base高度低于0.3m
# 20260214
## v0.0.1
1. 在go2_rl_gym基础上修改，加入unitree_mujoco中的g1_29dof.xml模型
2. 训练修改部分：
    1. 创建98维度的输入，全身关节输入，在go2基础上，最后两维加入sim,cos步频参数，周期为0.68s
    2. 模型文件配置，修改高度0.793m，base height为0.78m，控制PD，当躯干、手、头触碰地面终止，打开自碰撞，关闭视觉mesh翻转
    3. 加入`upper_body_to_default`奖励，鼓励上半身保持默认姿态
> 第一版先不加入步态奖励，测试学习效果
