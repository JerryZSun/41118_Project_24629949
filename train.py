import sys
sys.path.append('..')
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import simple_driving
import os
import math

# ========================================================
# Reward Constants
# ========================================================
SPEED_REWARD_SCALE      =  3.0
PROGRESS_REWARD_SCALE   = 15.0
OFF_TRACK_PENALTY       = -50.0
STEP_PENALTY            = -0.05
STEERING_SMOOTH_PENALTY = -2.5   # increased from -1.5


def custom_observation(client, car_pos, car_orn, speed, yaw, curr_s, track):
    x, y           = car_pos[0], car_pos[1]
    wall_distances = track.get_wall_distances(x, y, yaw, n_rays=7)
    norm_progress  = (curr_s % track.total_length) / track.total_length
    return wall_distances + [speed, norm_progress, 0.0]


def custom_reward(car_pos, car_orn, speed, progress, on_track,
                  lap_count, total_distance, steering_delta=0.0):
    """
    Reward for emergent racing line.
    - Strong speed + progress rewards drive the agent to go fast
    - Stronger steering smoothness penalty reduces wobble
    - No centreline constraint — agent free to find fastest path
    """
    if not on_track:
        return OFF_TRACK_PENALTY

    reward = 0.0

    if progress > 0:
        reward += progress * PROGRESS_REWARD_SCALE

    reward += speed * SPEED_REWARD_SCALE
    reward += steering_delta * STEERING_SMOOTH_PENALTY
    reward += STEP_PENALTY

    return reward


# ========================================================
# Training
# ========================================================
TOTAL_TIMESTEPS = 700000   # more steps for complex track + chicane
N_ENVS          = 4
MODEL_SAVE_PATH = "model/ppo_racing_model"

if __name__ == "__main__":
    os.makedirs("model", exist_ok=True)

    env_kwargs = {
        "renders":              False,
        "isDiscrete":           False,
        "reward_callback":      custom_reward,
        "observation_callback": custom_observation,
    }

    print(f"Creating {N_ENVS} parallel environments...")
    env = make_vec_env(
        "SimpleDriving-v0",
        n_envs=N_ENVS,
        vec_env_cls=SubprocVecEnv,
        env_kwargs=env_kwargs,
        vec_env_kwargs={"start_method": "spawn"}
    )

    # Uncomment to continue from checkpoint:
    # model = PPO.load(MODEL_SAVE_PATH, env=env)

    print("Instantiating PPO agent...")
    model = PPO(
        "MlpPolicy", env,
        learning_rate   = 0.0003,
        n_steps         = 1024,
        batch_size      = 256,
        ent_coef        = 0.01,
        verbose         = 1,
        tensorboard_log = "./ppo_tensorboard/"
    )

    checkpoint_callback = CheckpointCallback(
        save_freq   = max(50000 // N_ENVS, 1),
        save_path   = "./model/checkpoints/",
        name_prefix = "ppo_racing_checkpoint"
    )

    print(f"Training for {TOTAL_TIMESTEPS} timesteps...")
    print("Monitor: tensorboard --logdir=./ppo_tensorboard/")
    model.learn(
        total_timesteps = TOTAL_TIMESTEPS,
        callback        = checkpoint_callback,
        progress_bar    = True
    )
    model.save(MODEL_SAVE_PATH)
    print(f"Done! Model saved to {MODEL_SAVE_PATH}.zip")