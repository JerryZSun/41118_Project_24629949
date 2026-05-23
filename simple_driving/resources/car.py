import pybullet as p
import os
import math


class Car:
    def __init__(self, client, base_position=None, base_orientation=None):
        self.client = client
        f_name = os.path.join(os.path.dirname(__file__), 'simplecar.urdf')

        # Default to origin if not specified
        if base_position is None:
            base_position = [0, 0, 0.1]
        if base_orientation is None:
            base_orientation = [0, 0, 0, 1]

        self.car = self.client.loadURDF(
            fileName=f_name,
            basePosition=base_position,
            baseOrientation=base_orientation
        )

        # Joint indices as found by p.getJointInfo()
        self.steering_joints = [0, 2]
        self.drive_joints    = [1, 3, 4, 5]
        self.joint_speed     = 0

        # Drag constants
        self.c_rolling  = 0.2
        self.c_drag     = 0.01
        self.c_throttle = 100

    def get_ids(self):
        return self.car

    def apply_action(self, action):
        throttle, steering_angle = action
        throttle       = min(max(throttle, -1), 1)
        steering_angle = max(min(steering_angle, 0.6), -0.6)

        self.client.setJointMotorControlArray(
            self.car, self.steering_joints,
            controlMode=p.POSITION_CONTROL,
            targetPositions=[steering_angle] * 2
        )

        friction     = -self.joint_speed * (self.joint_speed * self.c_drag + self.c_rolling)
        acceleration = self.c_throttle * throttle + friction
        self.joint_speed = min(self.joint_speed + 0.01 * acceleration, 10.0)

        self.client.setJointMotorControlArray(
            bodyUniqueId=self.car,
            jointIndices=self.drive_joints,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocities=[self.joint_speed] * 4,
            forces=[1.2] * 4
        )

    def get_observation(self):
        pos, ang = self.client.getBasePositionAndOrientation(self.car)
        ang = p.getEulerFromQuaternion(ang)
        ori = (math.cos(ang[2]), math.sin(ang[2]))
        pos = pos[:2]
        vel = self.client.getBaseVelocity(self.car)[0][0:2]
        return pos + ori + vel