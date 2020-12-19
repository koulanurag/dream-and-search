# Consolidated Cassie environment.
try:
    from .cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis
except ImportError:
    from cassiemujoco import pd_in_t, state_out_t, CassieSim, CassieVis

try:
    from .udp import euler2quat, quaternion_product, inverse_quaternion, quaternion2euler, rotate_by_quaternion
except ImportError:
    from udp import euler2quat, quaternion_product, inverse_quaternion, quaternion2euler, rotate_by_quaternion

from math import floor

import numpy as np
import os
import random

import pickle
import time


class CassieEnv_v2:
    def __init__(self, traj='walking', simrate=60, dynamics_randomization=False, history=0, impedance=False, **kwargs):
        self.sim = CassieSim(os.environ['CASSIE_XML_PATH'])
        self.vis = None

        self.clock = True
        self.dynamics_randomization = dynamics_randomization
        self.state_est = True

        state_est_size = 35
        clock_size = 2  # [sin(t), cos(t)]
        speed_size = 2  # [x speed, y speed]
        height_size = 1  # [pelvis height, foot apex height]
        ratio_size = 1  # [ratio]

        obs_size = state_est_size + speed_size + height_size + clock_size + ratio_size

        self.observation_space = np.zeros(obs_size)

        # Adds option for state history for FF nets
        self._obs = len(self.observation_space)
        self.history = history

        self.observation_space = np.zeros(self._obs + self._obs * self.history)

        if impedance:
            self.action_space = np.zeros(30)
            # self.action_space = np.zeros(20)
        else:
            self.action_space = np.zeros(10)

        self.impedance = impedance

        self.phase_len = 1700

        self.P = np.array([100, 100, 88, 96, 50])
        self.D = np.array([10.0, 10.0, 8.0, 9.6, 5.0])

        self.u = pd_in_t()

        # TODO: should probably initialize this to current state
        self.cassie_state = state_out_t()
        self.simrate = simrate  # simulate X mujoco steps
        self.time = 0  # number of time steps in current episode
        self.phase = 0  # portion of the phase the robot is in
        self.counter = 0  # number of phase cycles completed in episode

        self.offset = np.array([0.0045, 0.0, 0.4973, -1.1997, -1.5968, 0.0045, 0.0, 0.4973, -1.1997, -1.5968])

        self.max_orient_change = 0.15

        self.max_speed = 3.4
        self.min_speed = -0.5

        self.max_side_speed = 0.3
        self.min_side_speed = -0.3

        self.max_step_freq = 1.5
        self.min_step_freq = 0.9

        self.max_height = 1.05
        self.min_height = 0.75

        # self.max_foot_height = 0.13
        # self.min_foot_height = 0.03

        self.max_pitch_incline = 0.03
        self.max_roll_incline = 0.03

        self.encoder_noise = 0.05

        self.damping_low = 0.3
        self.damping_high = 5.0

        self.max_simrate = 70
        self.min_simrate = 50

        self.mass_low = 0.5
        self.mass_high = 1.75

        self.fric_low = 0.35
        self.fric_high = 1.1

        self.speed = 0
        self.side_speed = 0
        self.orient_add = 0
        self.height = 1.0
        # self.foot_height = 0.05

        self.min_swing_ratio = 0.45
        self.max_swing_ratio = 0.7

        # Record default dynamics parameters
        self.default_damping = self.sim.get_dof_damping()
        self.default_mass = self.sim.get_body_mass()
        self.default_ipos = self.sim.get_body_ipos()
        self.default_fric = self.sim.get_geom_friction()
        self.default_rgba = self.sim.get_geom_rgba()
        self.default_quat = self.sim.get_geom_quat()
        self.default_simrate = simrate

        self.motor_encoder_noise = np.zeros(10)
        self.joint_encoder_noise = np.zeros(6)

    def step_simulation(self, action):

        self.sim_foot_frc.append(self.sim.get_foot_force())
        self.sim_height.append(self.cassie_state.pelvis.position[2] - self.cassie_state.terrain.height)

        target = action[:10] + self.offset
        p_add = np.zeros(10)
        d_add = np.zeros(10)

        if len(action) > 10:
            p_add = action[10:20]

        if len(action) > 20:
            d_add = action[20:30]

        if self.dynamics_randomization:
            target -= self.motor_encoder_noise

        self.u = pd_in_t()
        for i in range(5):
            # TODO: move setting gains out of the loop?
            # maybe write a wrapper for pd_in_t ?
            self.u.leftLeg.motorPd.pGain[i] = self.P[i] + p_add[i]
            self.u.rightLeg.motorPd.pGain[i] = self.P[i] + p_add[i + 5]

            self.u.leftLeg.motorPd.dGain[i] = self.D[i] + d_add[i]
            self.u.rightLeg.motorPd.dGain[i] = self.D[i] + d_add[i + 5]

            self.u.leftLeg.motorPd.torque[i] = 0  # Feedforward torque
            self.u.rightLeg.motorPd.torque[i] = 0

            self.u.leftLeg.motorPd.pTarget[i] = target[i]
            self.u.rightLeg.motorPd.pTarget[i] = target[i + 5]

            self.u.leftLeg.motorPd.dTarget[i] = 0
            self.u.rightLeg.motorPd.dTarget[i] = 0

        self.cassie_state = self.sim.step_pd(self.u)
        # if self.time > self.blackout_until:
        #    self.cassie_state = self.sim.step_pd(self.u)
        # else:
        #    self.sim.step_pd(self.u)

    def step(self, action):

        delay_rand = 4
        if self.dynamics_randomization:
            simrate = self.simrate + np.random.randint(-delay_rand, delay_rand + 1)
        else:
            simrate = self.simrate

        self.sim_foot_frc = []
        self.sim_height = []
        last_foot_pos = np.array(self.sim.foot_pos())
        for _ in range(simrate):
            self.step_simulation(action)
        self.foot_vel = (np.array(self.sim.foot_pos()) - last_foot_pos) / (simrate / 2000)

        self.time += 1
        self.phase += self.phase_add

        if self.phase >= self.phase_len:
            self.phase = self.phase % self.phase_len - 1
            self.counter += 1

        reward = self.compute_reward(action)
        done = self.done

        if np.random.randint(100) == 0:  # random changes to orientation
            self.current_turn_rate = np.random.uniform(-0.0075 * np.pi, 0.0075 * np.pi)
        if np.random.randint(100) == 0:
            self.current_turn_rate = 0
        self.orient_add += self.current_turn_rate

        if np.random.randint(300) == 0:  # random changes to commanded height
            self.height = np.random.uniform(self.min_height, self.max_height)

        # if np.random.randint(300) == 0: # random changes to commanded foot height
        #  self.foot_height = np.random.uniform(self.min_foot_height, self.max_foot_height)

        if np.random.randint(300) == 0:  # random changes to speed
            self.speed = np.random.uniform(self.min_speed, self.max_speed)

            # self.speed += np.random.uniform(-0.1, 0.5)
            # self.speed = np.clip(self.speed, self.min_speed, self.max_speed)
            # self.phase_add = int(self.simrate * self.bound_freq(self.speed, self.phase_add/self.simrate))
            # self.ratio     = self.bound_ratio(self.speed, ratio=self.ratio)

        if np.random.randint(300) == 0:  # random changes to sidespeed
            self.side_speed = np.random.uniform(self.min_side_speed, self.max_side_speed)

        if np.random.randint(300) == 0:  # random changes to clock speed
            self.phase_add = int(self.default_simrate * np.random.uniform(self.min_step_freq, self.max_step_freq))

            # self.phase_add = int(self.simrate * self.bound_freq(self.speed, generate_new=True))

        if np.random.randint(300) == 0:  # random changes to swing ratio
            self.ratio = np.random.uniform(self.min_swing_ratio, self.max_swing_ratio)

        if np.random.randint(300) == 0 and self.dynamics_randomization:
            self.simrate = int(np.random.uniform(self.min_simrate, self.max_simrate))

            # self.ratio = self.bound_ratio(self.speed)

        # if np.random.randint(100) == 0:
        #  self.blackout_until = self.time + np.random.randint(3, 9)

        state = self.get_full_state()

        self.last_action = action
        return state, reward, done, {}

    def rotate_to_orient(self, vec):
        quaternion = euler2quat(z=self.orient_add, y=0, x=0)
        iquaternion = inverse_quaternion(quaternion)

        if len(vec) == 3:
            return rotate_by_quaternion(vec, iquaternion)

        elif len(vec) == 4:
            new_orient = quaternion_product(iquaternion, vec)
            if new_orient[0] < 0:
                new_orient = -new_orient
            return new_orient

    def reset(self):
        self.phase = random.randint(0, self.phase_len)
        self.time = 0
        self.counter = 0
        # self.blackout_until = -1
        self.current_turn_rate = 0

        self.state_history = [np.zeros(self._obs) for _ in range(self.history + 1)]

        self.frc_sum = 0
        self.vel_sum = 0

        # Randomize dynamics:
        if self.dynamics_randomization:
            damp = self.default_damping

            pelvis_damp_range = [[damp[0], damp[0]],
                                 [damp[1], damp[1]],
                                 [damp[2], damp[2]],
                                 [damp[3], damp[3]],
                                 [damp[4], damp[4]],
                                 [damp[5], damp[5]]]  # 0->5

            hip_damp_range = [[damp[6] * self.damping_low, damp[6] * self.damping_high],
                              [damp[7] * self.damping_low, damp[7] * self.damping_high],
                              [damp[8] * self.damping_low, damp[8] * self.damping_high]]  # 6->8 and 19->21

            achilles_damp_range = [[damp[9] * self.damping_low, damp[9] * self.damping_high],
                                   [damp[10] * self.damping_low, damp[10] * self.damping_high],
                                   [damp[11] * self.damping_low, damp[11] * self.damping_high]]  # 9->11 and 22->24

            knee_damp_range = [[damp[12] * self.damping_low, damp[12] * self.damping_high]]  # 12 and 25
            shin_damp_range = [[damp[13] * self.damping_low, damp[13] * self.damping_high]]  # 13 and 26
            tarsus_damp_range = [[damp[14] * self.damping_low, damp[14] * self.damping_high]]  # 14 and 27

            heel_damp_range = [[damp[15], damp[15]]]  # 15 and 28
            fcrank_damp_range = [[damp[16] * self.damping_low, damp[16] * self.damping_high]]  # 16 and 29
            prod_damp_range = [[damp[17], damp[17]]]  # 17 and 30
            foot_damp_range = [[damp[18] * self.damping_low, damp[18] * self.damping_high]]  # 18 and 31

            side_damp = hip_damp_range + achilles_damp_range + knee_damp_range + shin_damp_range + tarsus_damp_range + heel_damp_range + fcrank_damp_range + prod_damp_range + foot_damp_range
            damp_range = pelvis_damp_range + side_damp + side_damp
            damp_noise = [np.random.uniform(a, b) for a, b in damp_range]

            m = self.default_mass
            pelvis_mass_range = [[self.mass_low * m[1], self.mass_high * m[1]]]  # 1
            hip_mass_range = [[self.mass_low * m[2], self.mass_high * m[2]],  # 2->4 and 14->16
                              [self.mass_low * m[3], self.mass_high * m[3]],
                              [self.mass_low * m[4], self.mass_high * m[4]]]

            achilles_mass_range = [[self.mass_low * m[5], self.mass_high * m[5]]]  # 5 and 17
            knee_mass_range = [[self.mass_low * m[6], self.mass_high * m[6]]]  # 6 and 18
            knee_spring_mass_range = [[self.mass_low * m[7], self.mass_high * m[7]]]  # 7 and 19
            shin_mass_range = [[self.mass_low * m[8], self.mass_high * m[8]]]  # 8 and 20
            tarsus_mass_range = [[self.mass_low * m[9], self.mass_high * m[9]]]  # 9 and 21
            heel_spring_mass_range = [[self.mass_low * m[10], self.mass_high * m[10]]]  # 10 and 22
            fcrank_mass_range = [[self.mass_low * m[11], self.mass_high * m[11]]]  # 11 and 23
            prod_mass_range = [[self.mass_low * m[12], self.mass_high * m[12]]]  # 12 and 24
            foot_mass_range = [[self.mass_low * m[13], self.mass_high * m[13]]]  # 13 and 25

            side_mass = hip_mass_range + achilles_mass_range \
                        + knee_mass_range + knee_spring_mass_range \
                        + shin_mass_range + tarsus_mass_range \
                        + heel_spring_mass_range + fcrank_mass_range \
                        + prod_mass_range + foot_mass_range

            mass_range = [[0, 0]] + pelvis_mass_range + side_mass + side_mass
            mass_noise = [np.random.uniform(a, b) for a, b in mass_range]

            delta = 0.0
            com_noise = [0, 0, 0] + [np.random.uniform(val - delta, val + delta) for val in self.default_ipos[3:]]

            fric_noise = []
            translational = np.random.uniform(self.fric_low, self.fric_high)
            torsional = np.random.uniform(1e-4, 5e-4)
            rolling = np.random.uniform(1e-4, 2e-4)
            for _ in range(int(len(self.default_fric) / 3)):
                fric_noise += [translational, torsional, rolling]

            geom_plane = [np.random.uniform(-self.max_roll_incline, self.max_roll_incline),
                          np.random.uniform(-self.max_pitch_incline, self.max_pitch_incline), 0]
            quat_plane = euler2quat(z=geom_plane[2], y=geom_plane[1], x=geom_plane[0])
            geom_quat = list(quat_plane) + list(self.default_quat[4:])

            self.motor_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=10)
            self.joint_encoder_noise = np.random.uniform(-self.encoder_noise, self.encoder_noise, size=6)

            self.sim.set_dof_damping(np.clip(damp_noise, 0, None))
            self.sim.set_body_mass(np.clip(mass_noise, 0, None))
            self.sim.set_body_ipos(com_noise)
            self.sim.set_geom_friction(np.clip(fric_noise, 0, None))
            self.sim.set_geom_quat(geom_quat)
            self.simrate = int(np.random.uniform(self.min_simrate, self.max_simrate))
        else:
            self.sim.set_body_mass(self.default_mass)
            self.sim.set_body_ipos(self.default_ipos)
            self.sim.set_dof_damping(self.default_damping)
            self.sim.set_geom_friction(self.default_fric)
            self.sim.set_geom_quat(self.default_quat)
            self.simrate = int(self.default_simrate)

            self.motor_encoder_noise = np.zeros(10)
            self.joint_encoder_noise = np.zeros(6)

        self.sim.set_const()

        self.cassie_state = self.sim.step_pd(self.u)

        self.orient_add = 0
        self.speed = np.random.uniform(-0.5, 1.0)
        self.side_speed = np.random.uniform(self.min_side_speed, self.max_side_speed)
        self.height = np.random.uniform(self.min_height, self.max_height)
        # self.foot_height = np.random.uniform(self.min_foot_height, self.max_foot_height)
        # self.phase_add   = int(self.simrate * self.bound_freq(self.speed, generate_new=True))
        self.phase_add = int(self.default_simrate * np.random.uniform(self.min_step_freq, self.max_step_freq))
        self.ratio = np.random.uniform(self.min_swing_ratio, self.max_swing_ratio)

        self.last_action = None
        self.last_left_pos = self.cassie_state.leftFoot.position[:]
        self.last_right_pos = self.cassie_state.rightFoot.position[:]

        self.last_t = self.sim.time()

        state = self.get_full_state()

        return state

    def compute_reward(self, action):
        #####################
        # HEIGHT COST TERMS #
        #####################

        sim_height = np.mean(self.sim_height)
        pelvis_hgt = np.abs(sim_height - self.height) * 3

        if pelvis_hgt < 0.02:
            pelvis_hgt = 0

        ####################
        # SPEED COST TERMS #
        ####################

        pelvis_vel = self.rotate_to_orient(self.cassie_state.pelvis.translationalVelocity[:])

        x_vel = np.abs(pelvis_vel[0] - self.speed)
        if x_vel < 0.05:
            x_vel = 0

        y_vel = np.abs(pelvis_vel[1] - self.side_speed)
        if y_vel < 0.05:
            y_vel = 0

        x_vel *= 4

        ##########################
        # ORIENTATION COST TERMS #
        ##########################

        actual_q = self.rotate_to_orient(self.cassie_state.pelvis.orientation[:])
        target_q = [1, 0, 0, 0]
        orientation_error = 6 * (1 - np.inner(actual_q, target_q) ** 2)

        left_actual = quaternion_product(actual_q, self.cassie_state.leftFoot.orientation)
        right_actual = quaternion_product(actual_q, self.cassie_state.rightFoot.orientation)

        left_actual_target_euler = quaternion2euler(left_actual) * [0, 1, 0]  # ROLL PITCH YAW
        right_actual_target_euler = quaternion2euler(right_actual) * [0, 1, 0]

        left_actual_target = euler2quat(z=left_actual_target_euler[2], y=left_actual_target_euler[1],
                                        x=left_actual_target_euler[0])
        right_actual_target = euler2quat(z=right_actual_target_euler[2], y=right_actual_target_euler[1],
                                         x=right_actual_target_euler[0])

        foot_err = 10 * ((1 - np.inner(left_actual, left_actual_target) ** 2) + (
                    1 - np.inner(right_actual, right_actual_target) ** 2))

        ######################
        # CLOCK REWARD TERMS #
        ######################

        ratio = self.ratio
        clock1_swing = self.reward_clock(ratio=ratio, alpha=0.15 * ratio, flip=False)
        clock1_stance = self.reward_clock(ratio=1 - ratio, alpha=0.15 * (1 - ratio), flip=True)

        clock2_swing = self.reward_clock(ratio=ratio, alpha=0.15 * ratio, flip=True)
        clock2_stance = self.reward_clock(ratio=1 - ratio, alpha=0.15 * (1 - ratio), flip=False)

        foot_frc = np.mean(self.sim_foot_frc, axis=0)
        left_frc = np.abs(foot_frc[0:3]).sum() / 100
        right_frc = np.abs(foot_frc[6:9]).sum() / 100

        # pelvis_speed = np.sqrt(np.power(pelvis_vel, 2).sum())

        # pelvis_vel     = np.array(self.cassie_state.pelvis.translationalVelocity[:])
        # left_foot_vel  = np.array(self.cassie_state.leftFoot.footTranslationalVelocity[:]) - pelvis_vel
        # right_foot_vel = np.array(self.cassie_state.rightFoot.footTranslationalVelocity[:]) - pelvis_vel

        left_vel = np.sqrt(np.power(self.foot_vel[:3], 2).sum())
        right_vel = np.sqrt(np.power(self.foot_vel[3:], 2).sum())

        # print("PELVIS VEL {:5.2}".format(pelvis_speed), end=', ')
        # if clock1_swing > 0:
        #  print("PUNISHING LEFT  FORCES {:3.2f} * {:5.2f}".format(clock1_swing, left_frc), end=", ")
        # else:
        #  print("IGNORING  LEFT  FORCES {:3.2f} * {:5.2f}".format(clock1_swing, left_frc), end=", ")

        # if clock2_swing > 0:
        #  print("PUNISHING RIGHT FORCES {:3.2f} * {:5.2f}".format(clock2_swing, right_frc), end=", ")
        # else:
        #  print("IGNORING  RIGHT FORCES {:3.2f} * {:5.2f}".format(clock2_swing, right_frc), end=", ")

        # if clock1_stance > 0:
        #  print("PUNISHING LEFT  VELOCITIES {:3.2f} * {:5.2f}".format(clock1_stance, left_vel), end=", ")
        # else:
        #  print("IGNORING  LEFT  VELOCITIES {:3.2f} * {:5.2f}".format(clock1_stance, left_vel), end=", ")

        # if clock2_stance > 0:
        #  print("PUNISHING RIGHT VELOCITIES {:3.2f} * {:5.2f}".format(clock2_stance, right_vel), end=", ")
        # else:
        #  print("IGNORING  RIGHT VELOCITIES {:3.2f} * {:5.2f}".format(clock2_stance, right_vel), end=", ")

        # Penalty which multiplies foot forces by 1 during swing, and 0 during stance.
        # (punish foot forces in the air)
        left_frc_penalty = np.abs(clock1_swing * left_frc)
        right_frc_penalty = np.abs(clock2_swing * right_frc)

        # Penalty which multiplies foot velocities by 1 during stance, and 0 during swing.
        # (punish foot velocities when foot is on the ground)
        left_vel_penalty = np.abs(clock1_stance * left_vel)
        right_vel_penalty = np.abs(clock2_stance * right_vel)

        # print("FRC PENALTY: {:6.3f}".format(np.exp(-(left_frc_penalty + right_frc_penalty))))
        # input()

        left_penalty = left_frc_penalty + left_vel_penalty
        right_penalty = right_frc_penalty + right_vel_penalty
        foot_frc_err = left_penalty + right_penalty

        # Penalty which punishes foot heights too far from the commanded apex
        lhgt = sim_height + self.cassie_state.leftFoot.position[:][2]
        rhgt = sim_height + self.cassie_state.rightFoot.position[:][2]

        # foot_height_err = 6 * (clock1_swing * np.abs(lhgt - self.foot_height) + \
        #                       clock2_swing * np.abs(rhgt - self.foot_height))

        ########################
        # JERKINESS COST TERMS #
        ########################

        # Torque cost term
        torque = np.asarray(self.cassie_state.motor.torque[:])
        torque_penalty = 0.05 * sum(np.abs(torque) / len(torque))

        # Action cost term
        if self.last_action is None:
            ctrl_penalty = 0
        else:
            ctrl_penalty = 5 * sum(np.abs(self.last_action - action)) / len(action)

        pelvis_acc = 0.15 * (np.abs(self.cassie_state.pelvis.rotationalVelocity[:]).sum() + np.abs(
            self.cassie_state.pelvis.translationalAcceleration[:]).sum())

        self.done = False
        if np.exp(-orientation_error) < 0.8:
            self.done = True

        reward = 0.000 + \
                 0.250 * np.exp(-(orientation_error + foot_err)) + \
                 0.200 * np.exp(-foot_frc_err) + \
                 0.200 * np.exp(-x_vel) + \
                 0.125 * np.exp(-pelvis_acc) + \
                 0.100 * np.exp(-y_vel) + \
                 0.075 * np.exp(-pelvis_hgt) + \
                 0.025 * np.exp(-ctrl_penalty) + \
                 0.025 * np.exp(-torque_penalty)

        return reward

    def get_friction(self):
        return np.hstack([self.sim.get_geom_friction()[0]])

    def get_damping(self):
        return np.hstack([self.sim.get_dof_damping()])

    def get_mass(self):
        return np.hstack([self.sim.get_body_mass()])

    def get_quat(self):
        return np.hstack([quaternion2euler(self.sim.get_geom_quat())[:2]])

    def get_clock(self):
        return [np.sin(2 * np.pi * self.phase / self.phase_len),
                np.cos(2 * np.pi * self.phase / self.phase_len)]

    def reward_clock(self, ratio=0.5, alpha=0.5, flip=False):
        phi = self.phase / self.phase_len
        beta = 0.0
        if flip:
            beta = 0.5
        phi = np.fmod(phi + beta, 1)

        saturation = alpha * (ratio / 2 - 1e-3)
        slope = 1 / ((ratio / 2) - saturation)

        if phi < saturation:
            return 1.0
        elif phi < ratio / 2:
            return 1 - slope * (phi - saturation)
        elif phi < 1 - ratio / 2:
            return 0.0
        elif phi < 1 - saturation:
            return 1 + slope * (phi + saturation - 1)
        else:
            return 1.0

    def bound_freq(self, speed, freq=None, generate_new=False):
        lower = np.interp(np.abs(speed), (0, 3), (0.9, 1.5))
        upper = np.interp(np.abs(speed), (2, 3), (1.5, 1.7))

        if generate_new:
            freq = np.random.uniform(lower, upper)
        elif freq is None:
            freq = self.phase_add / self.default_simrate
        freq = np.clip(freq, lower, upper)

        return freq

    def bound_ratio(self, speed, ratio=None):
        lower = np.interp(np.abs(speed), (0, 2), (self.min_swing_ratio, self.max_swing_ratio))
        upper = self.max_swing_ratio

        if ratio is None:
            ratio = np.random.uniform(lower, upper)

        ratio = np.clip(ratio, lower, upper)

        return ratio

    def get_full_state(self):
        qpos = np.copy(self.sim.qpos())
        qvel = np.copy(self.sim.qvel())

        clock = self.get_clock()

        ext_state = np.concatenate((clock, [self.speed, self.side_speed, self.height, self.ratio]))

        pelvis_quat = self.rotate_to_orient(self.cassie_state.pelvis.orientation)

        pelvis_vel = self.rotate_to_orient(self.cassie_state.pelvis.translationalVelocity[:])
        pelvis_rvel = self.cassie_state.pelvis.rotationalVelocity[:]

        if self.dynamics_randomization:
            motor_pos = self.cassie_state.motor.position[:] + self.motor_encoder_noise
            joint_pos = self.cassie_state.joint.position[:] + self.joint_encoder_noise
        else:
            motor_pos = self.cassie_state.motor.position[:]
            joint_pos = self.cassie_state.joint.position[:]

        motor_vel = self.cassie_state.motor.velocity[:]
        joint_vel = self.cassie_state.joint.velocity[:]

        # remove double-counted joint/motor positions
        joint_pos = np.concatenate([joint_pos[:2], joint_pos[3:5]])
        joint_vel = np.concatenate([joint_vel[:2], joint_vel[3:5]])

        robot_state = np.concatenate([
            pelvis_quat[:],  # pelvis orientation
            pelvis_rvel,  # pelvis rotational velocity
            motor_pos,  # actuated joint positions
            motor_vel,  # actuated joint velocities
            joint_pos,  # unactuated joint positions
            joint_vel  # unactuated joint velocities
        ])

        state = np.concatenate([robot_state, ext_state])

        self.state_history.insert(0, state)
        self.state_history = self.state_history[:self.history + 1]

        return np.concatenate(self.state_history)

    def render(self):
        if self.vis is None:
            self.vis = CassieVis(self.sim, os.environ['CASSIE_XML_PATH'])

        return self.vis.draw(self.sim)

    def mirror_state(self, state):
        state_est_indices = [0.01, 1, 2, 3,  # pelvis orientation
                             -4, 5, -6,  # rotational vel
                             -12, -13, 14, 15, 16,  # left motor pos
                             -7, -8, 9, 10, 11,  # right motor pos
                             -22, -23, 24, 25, 26,  # left motor vel
                             -17, -18, 19, 20, 21,  # right motor vel
                             29, 30, 27, 28,  # joint pos
                             33, 34, 31, 32, ]  # joint vel

        # state_est_indices = [0.01, 1, 2, 3,            # pelvis orientation
        #                     -9, -10, 11, 12, 13,      # left motor pos
        #                     -4,  -5,  6,  7,  8,      # right motor pos
        #                     14, -15, 16,              # translational vel
        #                     -17, 18, -19,             # rotational vel
        #                     -25, -26, 27, 28, 29,     # left motor vel
        #                     -20, -21, 22, 23, 24,     # right motor vel
        #                     32, 33, 30, 31,           # joint pos
        #                     36, 37, 34, 35, ]         # joint vel

        return_as_1d = False
        if isinstance(state, list):
            return_as_1d = True
            statedim = len(state)
            batchdim = 1
            state = np.asarray(state).reshape(1, -1)

        elif isinstance(state, np.ndarray):
            if len(state.shape) == 1:
                return_as_1d = True
                state = np.asarray(state).reshape(1, -1)

            statedim = state.shape[-1]
            batchdim = state.shape[0]

        else:
            raise NotImplementedError

        if statedim == len(state_est_indices) + 6:  # state estimator with clock and speed or height
            mirror_obs = state_est_indices + [len(state_est_indices) + i for i in range(6)]
            sidespeed = mirror_obs[-3]
            sinclock = mirror_obs[-5]
            cosclock = mirror_obs[-6]

            new_orient = state[:, :4]
            new_orient = np.array(list(map(inverse_quaternion, [new_orient[i] for i in range(batchdim)])))
            new_orient[:, 2] *= -1

            # if new_orient[:,0] < 0:
            #  new_orient = [-1 * x for x in new_orient]

            mirrored_state = np.copy(state)
            for idx, i in enumerate(mirror_obs):
                if i == sidespeed:
                    mirrored_state[:, idx] = -1 * state[:, idx]
                elif i == sinclock or i == cosclock:
                    mirrored_state[:, idx] = (np.sin(np.arcsin(state[:, i]) + np.pi))
                else:
                    mirrored_state[:, idx] = (np.sign(i) * state[:, abs(int(i))])

            mirrored_state = np.hstack([new_orient, mirrored_state[:, 4:]])
            if return_as_1d:
                return np.asarray(mirrored_state)[0]
            else:
                return np.asarray(mirrored_state)

        else:
            raise RuntimeError

    def mirror_action(self, action):
        return_as_1d = False
        if isinstance(action, list):
            return_as_1d = True
            actiondim = len(state)
            batchdim = 1
            action = np.asarray(state).reshape(1, -1)

        elif isinstance(action, np.ndarray):
            if len(action.shape) == 1:
                return_as_1d = True
                action = np.asarray(action).reshape(1, -1)

            actiondim = action.shape[-1]
            batchdim = action.shape[0]

        else:
            raise NotImplementedError
        mirror_act = np.copy(action)

        idxs = [-5, -6, 7, 8, 9,
                -0.1, -1, 2, 3, 4]

        if actiondim > 10:
            idxs += [-15, -16, 17, 18, 19,
                     -10, -11, 12, 13, 14]

        if actiondim > 20:
            idxs += [-25, -26, 27, 28, 29,
                     -20, -21, 22, 23, 24]

        for idx, i in enumerate(idxs):
            mirror_act[:, idx] = (np.sign(i) * action[:, abs(int(i))])
        if return_as_1d:
            return mirror_act.reshape(-1)
        else:
            return mirror_act

# nbody layout:
# 0:  worldbody (zero)
# 1:  pelvis

# 2:  left hip roll 
# 3:  left hip yaw
# 4:  left hip pitch
# 5:  left achilles rod
# 6:  left knee
# 7:  left knee spring
# 8:  left shin
# 9:  left tarsus
# 10:  left heel spring
# 12:  left foot crank
# 12: left plantar rod
# 13: left foot

# 14: right hip roll 
# 15: right hip yaw
# 16: right hip pitch
# 17: right achilles rod
# 18: right knee
# 19: right knee spring
# 20: right shin
# 21: right tarsus
# 22: right heel spring
# 23: right foot crank
# 24: right plantar rod
# 25: right foot


# qpos layout
# [ 0] Pelvis x
# [ 1] Pelvis y
# [ 2] Pelvis z
# [ 3] Pelvis orientation qw
# [ 4] Pelvis orientation qx
# [ 5] Pelvis orientation qy
# [ 6] Pelvis orientation qz
# [ 7] Left hip roll         (Motor [0])
# [ 8] Left hip yaw          (Motor [1])
# [ 9] Left hip pitch        (Motor [2])
# [10] Left achilles rod qw
# [11] Left achilles rod qx
# [12] Left achilles rod qy
# [13] Left achilles rod qz
# [14] Left knee             (Motor [3])
# [15] Left shin                        (Joint [0])
# [16] Left tarsus                      (Joint [1])
# [17] Left heel spring
# [18] Left foot crank
# [19] Left plantar rod
# [20] Left foot             (Motor [4], Joint [2])
# [21] Right hip roll        (Motor [5])
# [22] Right hip yaw         (Motor [6])
# [23] Right hip pitch       (Motor [7])
# [24] Right achilles rod qw
# [25] Right achilles rod qx
# [26] Right achilles rod qy
# [27] Right achilles rod qz
# [28] Right knee            (Motor [8])
# [29] Right shin                       (Joint [3])
# [30] Right tarsus                     (Joint [4])
# [31] Right heel spring
# [32] Right foot crank
# [33] Right plantar rod
# [34] Right foot            (Motor [9], Joint [5])

# qvel layout
# [ 0] Pelvis x
# [ 1] Pelvis y
# [ 2] Pelvis z
# [ 3] Pelvis orientation wx
# [ 4] Pelvis orientation wy
# [ 5] Pelvis orientation wz
# [ 6] Left hip roll         (Motor [0])
# [ 7] Left hip yaw          (Motor [1])
# [ 8] Left hip pitch        (Motor [2])
# [ 9] Left knee             (Motor [3])
# [10] Left shin                        (Joint [0])
# [11] Left tarsus                      (Joint [1])
# [12] Left foot             (Motor [4], Joint [2])
# [13] Right hip roll        (Motor [5])
# [14] Right hip yaw         (Motor [6])
# [15] Right hip pitch       (Motor [7])
# [16] Right knee            (Motor [8])
# [17] Right shin                       (Joint [3])
