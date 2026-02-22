from pathlib import Path
import matplotlib.pyplot as plt

PATH_LOGS = Path(__file__).parents[1] / "logs"
PATH_LOGS.mkdir(exist_ok=True)

reward_str = """
        Mean episode rew_collision: -0.0018
 Mean episode rew_tracking_lin_vel: 0.3576
          Mean episode rew_dof_acc: -0.0447
  Mean episode rew_feet_regulation: -0.0021
Mean episode rew_upper_body_to_default: -0.0638
        Mean episode rew_dof_power: -0.0024
        Mean episode rew_lin_vel_z: 0.0000
    Mean episode rew_parallel_feet: -0.0011
       Mean episode rew_ang_vel_xy: -0.0220
Mean episode rew_correct_base_height: -0.0170
Mean episode rew_stance_to_default: -0.0135
   Mean episode rew_dof_pos_limits: -0.0042
      Mean episode rew_action_rate: -0.0466
 Mean episode rew_tracking_ang_vel: 0.1092
   Mean episode rew_orientation_xy: -0.0061
Mean episode rew_action_smoothness: -0.0899
          Mean episode rew_torques: -0.0828
"""
skip_rewards = ['tracking_ang_vel', 'tracking_lin_vel']

if __name__ == '__main__':
    name2value = {}
    for line in reward_str.splitlines():
        if line.strip():
            name, value = line.split(":")
            name = name.strip()
            name = name.split("Mean episode ")[-1]
            name = name.split("rew_")[-1]
            if name in skip_rewards:
                continue
            value = float(value.strip())
            name2value[name] = value
            
    name2value = dict(sorted(name2value.items(), key=lambda item: item[0]))
    
    for name, value in name2value.items():
        plt.barh(name, value)
    
    plt.grid(axis='both', linestyle='--', alpha=0.7)
    plt.savefig(PATH_LOGS / "reward_bins.png", bbox_inches='tight')
    plt.close()
    print(f"{PATH_LOGS / 'reward_bins.png'} saved.")