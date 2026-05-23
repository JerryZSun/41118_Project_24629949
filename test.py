import sys
sys.path.append('..')
import gymnasium as gym
from stable_baselines3 import PPO
import simple_driving
import time
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from simple_driving.resources.track import (
    Track, TRACK_WIDTH, CORNER_RADIUS, STRAIGHT_LEN,
    INNER_A, INNER_B, build_centreline
)
from train import custom_reward, custom_observation

OUTPUT_DIR = r"C:\Users\User\Desktop\Project 24629949\output"


def draw_track_boundary(ax):
    OR = CORNER_RADIUS + TRACK_WIDTH / 2.0   # outer wall radius
    hw = TRACK_WIDTH / 2.0

    # --- Outer wall (rounded rectangle) ---
    ax.plot([-STRAIGHT_LEN, STRAIGHT_LEN], [-OR, -OR],
            color='red', linewidth=2, label='Outer wall')
    ax.plot([-STRAIGHT_LEN, STRAIGHT_LEN], [OR, OR],
            color='red', linewidth=2)
    t = np.linspace(-math.pi/2, math.pi/2, 120)
    ax.plot(STRAIGHT_LEN  + OR*np.cos(t), OR*np.sin(t), color='red', linewidth=2)
    t = np.linspace(math.pi/2, 3*math.pi/2, 120)
    ax.plot(-STRAIGHT_LEN + OR*np.cos(t), OR*np.sin(t), color='red', linewidth=2)

    # --- Inner wall (ellipse) ---
    theta = np.linspace(0, 2*math.pi, 400)
    ax.plot(INNER_A*np.cos(theta), INNER_B*np.sin(theta),
            color='blue', linewidth=2, label='Inner wall')

    # --- Centreline ---
    cl   = build_centreline(300)
    cl_c = np.vstack([cl, cl[0]])
    ax.plot(cl_c[:, 0], cl_c[:, 1],
            color='gray', linewidth=1, linestyle='--', alpha=0.4, label='Centreline')

    # --- Track surface fill ---
    t_out = np.linspace(-math.pi/2, math.pi/2, 80)
    outer_x = np.concatenate([
        np.linspace(-STRAIGHT_LEN,  STRAIGHT_LEN, 80),
        STRAIGHT_LEN  + OR*np.cos(t_out),
        np.linspace( STRAIGHT_LEN, -STRAIGHT_LEN, 80),
        -STRAIGHT_LEN + OR*np.cos(t_out + math.pi)
    ])
    outer_y = np.concatenate([
        np.full(80, -OR),
        OR*np.sin(t_out),
        np.full(80,  OR),
        OR*np.sin(t_out + math.pi)
    ])
    ax.fill(outer_x, outer_y,
            color='#444444', alpha=0.2, zorder=0)
    ax.fill(INNER_A*np.cos(theta), INNER_B*np.sin(theta),
            color='white', alpha=1.0, zorder=1)

    # --- Start/finish ---
    sx, sy, _ = Track.get_start_position()
    ax.plot([sx, sx], [sy - hw, sy + hw],
            color='green', linewidth=3, label='Start/Finish', zorder=3)


def plot_episode_path(path, episode_num, total_reward, laps, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(16, 9))
    draw_track_boundary(ax)

    if len(path) > 1:
        xs = [pt[0] for pt in path]
        ys = [pt[1] for pt in path]
        n  = len(xs)
        for i in range(n - 1):
            t     = i / max(n - 2, 1)
            color = plt.cm.plasma(t)
            ax.plot(xs[i:i+2], ys[i:i+2], color=color,
                    linewidth=2.5, alpha=0.9, zorder=4)
        ax.scatter(xs[0],  ys[0],  color='lime',   s=120, zorder=5, label='Start')
        ax.scatter(xs[-1], ys[-1], color='orange',  s=120, zorder=5,
                   marker='X', label='End')

    OR     = CORNER_RADIUS + TRACK_WIDTH / 2.0
    margin = 3
    ax.set_xlim(-(STRAIGHT_LEN + OR + margin), (STRAIGHT_LEN + OR + margin))
    ax.set_ylim(-(OR + margin), (OR + margin))
    ax.set_aspect('equal')
    ax.set_title(
        f'Episode {episode_num} — Driven Path\n'
        f'Reward: {total_reward:.1f}  |  Laps: {laps}  |  Points: {len(path)}',
        fontsize=13
    )
    ax.set_xlabel('X (metres)')
    ax.set_ylabel('Y (metres)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.2)

    sm   = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.6)
    cbar.set_label('Episode progress  (early → late)', fontsize=9)

    fname = os.path.join(save_dir, f"episode_{episode_num}_path.png")
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved → {fname}")


def test_policy(n_episodes=3):
    print("Loading environment...")
    env = gym.make(
        "SimpleDriving-v0",
        renders=True,
        isDiscrete=False,
        reward_callback=custom_reward,
        observation_callback=custom_observation
    )

    print("Loading model...")
    model    = PPO.load("model/ppo_racing_model", env=env)
    base_env = env.unwrapped

    for ep in range(n_episodes):
        print(f"\n--- Episode {ep + 1} ---")
        obs, _         = env.reset()
        done           = False
        episode_reward = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            time.sleep(0.01)

        laps = base_env.lap_count
        path = base_env.get_path_history()
        print(f"  Reward : {episode_reward:.2f}")
        print(f"  Laps   : {laps}")
        print(f"  Points : {len(path)}")
        plot_episode_path(path, ep + 1, episode_reward, laps, OUTPUT_DIR)

    env.close()
    print(f"\nAll images saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    test_policy()