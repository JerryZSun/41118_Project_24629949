import gymnasium as gym
import numpy as np
import math
import pybullet as p
from pybullet_utils import bullet_client as bc
from simple_driving.resources.car import Car
from simple_driving.resources.plane import Plane
from simple_driving.resources.track import Track
import time
from collections import deque

RENDER_HEIGHT = 720
RENDER_WIDTH  = 960
N_RAYS        = 7
OBS_HISTORY   = 2   # number of steps to average observations over


class SimpleDrivingEnv(gym.Env):
    metadata = {'render_modes': ['human', 'tp_camera', 'rgb_array']}

    def __init__(self, isDiscrete=False, renders=False,
                 reward_callback=None, observation_callback=None):

        self.action_space = gym.spaces.box.Box(
            low=np.array([-1, -.6], dtype=np.float32),
            high=np.array([ 1,  .6], dtype=np.float32)
        )

        obs_size = N_RAYS + 3   # rays + speed + norm_progress + smoothed_steering
        self.observation_space = gym.spaces.box.Box(
            low =np.full(obs_size, -1.0, dtype=np.float32),
            high=np.full(obs_size, 10.0, dtype=np.float32)
        )

        self.np_random, _ = gym.utils.seeding.np_random()

        if renders:
            self._p = bc.BulletClient(connection_mode=p.GUI)
        else:
            self._p = bc.BulletClient()

        self._renders        = renders
        self._timeStep       = 0.01
        self._actionRepeat   = 20
        self._envStepCounter = 0

        self.car   = None
        self.track = None
        self.done  = False

        self.prev_s            = 0.0
        self.total_distance    = 0.0
        self.lap_count         = 0
        self.path_history      = []

        # Steering smoothing — heavier alpha reduces wobble
        self.smoothed_steering = 0.0
        self.SMOOTH_ALPHA      = 0.6   # increased from 0.4

        # Rolling average over last OBS_HISTORY ray observations
        # smooths out flickering ray distances and reduces reactive steering
        self._ray_buffer = deque(maxlen=OBS_HISTORY)

        self.reward_callback      = reward_callback
        self.observation_callback = observation_callback

    # ------------------------------------------------------------------
    def step(self, action):
        throttle, raw_steering = action

        # Exponential moving average — car physically cannot steer instantly
        self.smoothed_steering = (self.SMOOTH_ALPHA * self.smoothed_steering
                                  + (1 - self.SMOOTH_ALPHA) * raw_steering)
        self.car.apply_action([throttle, self.smoothed_steering])

        for _ in range(self._actionRepeat):
            self._p.stepSimulation()
            if self._renders:
                time.sleep(self._timeStep)
            if self._termination():
                self.done = True
                break
            self._envStepCounter += 1

        car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
        x, y = car_pos[0], car_pos[1]
        self.path_history.append((x, y))

        on_track = self.track.point_on_track(x, y)
        if not on_track:
            self.done = True

        curr_s = self.track.get_progress(x, y)
        delta  = self.track.progress_delta(self.prev_s, curr_s)
        if delta > 0:
            self.total_distance += delta
        self.prev_s = curr_s

        new_laps = self.track.laps_completed(self.total_distance)
        if new_laps > self.lap_count:
            self.lap_count = new_laps

        vel, _ = self._p.getBaseVelocity(self.car.car)
        speed  = math.sqrt(vel[0]**2 + vel[1]**2)
        euler  = self._p.getEulerFromQuaternion(car_orn)
        yaw    = euler[2]

        steering_delta = abs(raw_steering - self.smoothed_steering)
        ob = self._get_observation(car_pos, car_orn, speed, yaw, curr_s)

        if self.reward_callback is None:
            raise ValueError("No reward_callback provided!")

        reward = self.reward_callback(
            car_pos        = car_pos,
            car_orn        = car_orn,
            speed          = speed,
            progress       = delta,
            on_track       = on_track,
            lap_count      = self.lap_count,
            total_distance = self.total_distance,
            steering_delta = steering_delta,
        )

        return np.array(ob, dtype=np.float32), float(reward), self.done, False, {}

    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._p.resetSimulation()
        self._p.setTimeStep(self._timeStep)
        self._p.setGravity(0, 0, -10)

        Plane(self._p)
        self.track = Track(self._p)

        sx, sy, syaw = Track.get_start_position()
        self.car = Car(
            self._p,
            base_position=[sx, sy, 0.1],
            base_orientation=self._p.getQuaternionFromEuler([0, 0, syaw])
        )

        self._envStepCounter   = 0
        self.done              = False
        self.lap_count         = 0
        self.total_distance    = 0.0
        self.path_history      = [(sx, sy)]
        self.smoothed_steering = 0.0
        self._ray_buffer.clear()

        car_pos, car_orn = self._p.getBasePositionAndOrientation(self.car.car)
        self.prev_s = self.track.get_progress(car_pos[0], car_pos[1])

        vel, _ = self._p.getBaseVelocity(self.car.car)
        speed  = math.sqrt(vel[0]**2 + vel[1]**2)
        euler  = self._p.getEulerFromQuaternion(car_orn)
        yaw    = euler[2]

        ob = self._get_observation(car_pos, car_orn, speed, yaw, self.prev_s)
        return np.array(ob, dtype=np.float32), {}

    # ------------------------------------------------------------------
    def _get_observation(self, car_pos, car_orn, speed, yaw, curr_s):
        if self.observation_callback is not None:
            return self.observation_callback(
                client=self._p, car_pos=car_pos, car_orn=car_orn,
                speed=speed, yaw=yaw, curr_s=curr_s, track=self.track,
            )

        x, y = car_pos[0], car_pos[1]

        # Raw ray distances
        raw_rays = self.track.get_wall_distances(x, y, yaw, n_rays=N_RAYS)

        # Rolling average over last OBS_HISTORY steps — smooths flickering
        self._ray_buffer.append(raw_rays)
        smoothed_rays = np.mean(list(self._ray_buffer), axis=0).tolist()

        norm_progress = (curr_s % self.track.total_length) / self.track.total_length

        return smoothed_rays + [speed, norm_progress, self.smoothed_steering]

    # ------------------------------------------------------------------
    def get_path_history(self):
        return list(self.path_history)

    def _termination(self):
        return self._envStepCounter > 12000

    def close(self):
        self._p.disconnect()

    def render(self, mode='tp_camera'):
        if not self._renders:
            return np.array([])
        view = self._p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0],
            distance=40.0, yaw=0.0, pitch=-70, roll=0, upAxisIndex=2
        )
        proj = self._p.computeProjectionMatrixFOV(
            fov=60, aspect=float(RENDER_WIDTH)/RENDER_HEIGHT,
            nearVal=0.1, farVal=100.0
        )
        (_, _, px, _, _) = self._p.getCameraImage(
            width=RENDER_WIDTH, height=RENDER_HEIGHT,
            viewMatrix=view, projectionMatrix=proj,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        return np.array(px)[:, :, :3]