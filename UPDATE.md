# 20260222
## v0.0.4.5
v0.0.4.4的训练中没有`orientation_xy`中2个的一个训出来前后正常抬脚了，但是加入orientation_xy都存在问题
1. 降低`orientation_xy`奖励`-5.0 -> -2.0`
2. 提高`parallel_feet`奖励`-0.1 -> -0.2`
3. `correct_base_height`还原依旧最多增长到10
4. 模仿`feet_regulation`奖励设计`feet_diff_height`差异足端高度，奖励系数为`-0.05`相同，sigma系数更小为`0.007`，两个足端用一个表示，只考虑较大的一个足端速度$\max(||v_{xy}^{feet_l}||_2^2, ||v_{xy}^{feet_r}||_2^2)\exp\left(-\frac{|p_z^{feet_l}-p_z^{feet_r}|}{0.007h^{des}_{base}}\right)$

训练消融：
1. v0.0.4.5: 包含orientation_xy, parallel_feet, feet_diff_height
2. v0.0.4.5_no_orient: 包含parallel_feet, feet_diff_height
3. v0.0.4.5_no_diff：包含orientation_xy, parallel_feet
4. v0.0.4.5_no_all：不包含orientation_xy, feet_diff_height
# 20260220
## v0.0.4.4
v0.0.4.3直接传入步频训练效果很差，首先feet_regulation崩了，尝试直接用base_height引导
1. 修改`feet_regulation`可能是存在差异性高度问题
2. 修改`correct_base_height`线性增长从`0->1w`从`1->20`
## v0.0.4.3
v0.0.4.2问题在于向后移动时候都是跳跃姿势，考虑是否能加入步态的引导奖励，而非直接传入步频
1. 参考CTS加入新奖励`orientation_xy`保证上肢保持竖直
2. （先不加）两脚间距惩罚`feet_distance`，鼓励两脚间距大于阈值0.2m（先不添加这个，可能有问题，只在前后移动有效）
3. 修改`feet_regulation`奖励中计算脚距离地面高度为高度图方法，比之前投影的方法更精确，并加入双脚差异性高度条件，当有差异抬脚高度超过0.02m时才能认为是抬脚，避免双脚起跳的问题
4. 删除gait_phase观测项
# 20260219
## v0.0.4.2
v0.0.4.1依旧三个种子训出了一个不错的，一个向前时候蹦蹦跳跳，两个抬脚的，一个比较稳定，但是都有内八的问题，加入脚尖平齐奖励
1. 加入新奖励`parallel_feet=-0.1`，计算两脚roll中x向量在另一个脚的坐标系向xy平面投影和x的夹角，当夹角大于0.1rad，并且乘上sigma=0.1的z角速度指令系数，较大角速度执行时（>0.4rad/s）允许有一定的八字出现
# 20260218
## v0.0.4.1
0.0.4问题依旧有点后仰
1. 提高`upper_body_to_default`: `-0.05 -> -0.1`
2. `stance_body_to_default`: `-0.1 -> -0.2`
# 20260217
## v0.0.4
sim2sim发现几个问题：上肢还是有点后仰，发现上肢奖励写错了；静止时会来回晃动，修改静止指令并加入静止奖励

1. 修复上半身奖励还没使用upper_body_to_default配置的bug
2. 加入静止站立关节惩罚，鼓励机器人保持完全静止的站立姿态
3. 将原来的zero command改成完全指令静止，包括角速度指令（原来只有线速度为0）
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
