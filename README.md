# Go-Kart Racing Line Optimisation using Reinforcement Learning

A reinforcement learning project that trains a simulated go-kart to discover an optimal racing line using PPO (Proximal Policy Optimization) in a custom PyBullet environment.

The agent learns to navigate a closed circuit track using only wall distance sensors and speed feedback, without being told the correct path. Training is monitored via TensorBoard and driven path visualisations are generated after each test run.

**Subject:** 41118 AI for Robotics — University of Technology Sydney  
**Author:** Jerry Sun (24629949)

---

## Installation

### Step 1 — Install Python 3.11

PyBullet requires Python 3.11. Download from:  
https://www.python.org/downloads/release/python-3119/

During installation check **"Add python.exe to PATH"**.

### Step 2 — Install Microsoft C++ Build Tools

Required to compile PyBullet on Windows. Download from:  
https://visualstudio.microsoft.com/visual-cpp-build-tools/

When the installer opens check **"Desktop development with C++"**.

### Step 3 — Create and activate a virtual environment

```bash
python -m venv venv
```

```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\venv\Scripts\Activate.ps1
```

### Step 4 — Install dependencies

```bash
pip install gymnasium pybullet stable-baselines3[extra] numpy matplotlib
```

---

## Training

To start a fresh training run:

```bash
python train.py
```

To continue training from a saved checkpoint (curriculum learning), uncomment this line in `train.py`:

```python
# model = PPO.load(MODEL_SAVE_PATH, env=env)
```

To delete a previous model before retraining:

```bash
Remove-Item model\ppo_racing_model.zip
```

---

## Monitoring with TensorBoard

Open a second terminal, activate the venv, then run:

```bash
.\venv\Scripts\Activate.ps1
tensorboard --logdir=./ppo_tensorboard/
```

Open in browser: http://localhost:6006/

Watch **ep_rew_mean** under the rollout section. The reward should climb from negative toward positive as the agent learns to complete laps.

---

## Testing

Once training is complete:

```bash
.\venv\Scripts\Activate.ps1
python test.py
```

This runs 3 test episodes and saves a top-down path visualisation image for each episode to the `output/` folder. The path is coloured from dark purple (start) to yellow (end of episode).

---

## Expected Output

The agent should complete laps around the circuit. In the path visualisation, look for the driven line deviating from the centreline through corners — wider on entry, tighter at the apex, and wider again on exit. This is the optimal racing line the agent discovers through reward shaping alone.