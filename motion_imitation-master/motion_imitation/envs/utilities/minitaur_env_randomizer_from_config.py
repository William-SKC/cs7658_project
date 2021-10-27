# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python2, python3
"""An environment randomizer that randomizes physical parameters from config."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import functools
import random
import numpy as np
import six
import tensorflow.compat.v1 as tf
from motion_imitation.envs.utilities import env_randomizer_base
from motion_imitation.envs.utilities import minitaur_env_randomizer_config

SIMULATION_TIME_STEP = 0.001


class MinitaurEnvRandomizerFromConfig(env_randomizer_base.EnvRandomizerBase):
  """A randomizer that change the minitaur_gym_env during every reset."""
  def __init__(self, config=None):
    if config is None:
      config = "all_params"
    try:
      config = getattr(minitaur_env_randomizer_config, config)
    except AttributeError:
      raise ValueError("Config {} is not found.".format(config))
    self._randomization_param_dict = config()
    tf.logging.info("Randomization config is: {}".format(
        self._randomization_param_dict))

  def randomize_env(self, env):
    """Randomize various physical properties of the environment.

    It randomizes the physical parameters according to the input configuration.

    Args:
      env: A minitaur gym environment.
    """
    self._randomization_function_dict = self._build_randomization_function_dict(
        env)
    for param_name, random_range in six.iteritems(
        self._randomization_param_dict):
      self._randomization_function_dict[param_name](
          lower_bound=random_range[0], upper_bound=random_range[1])

  def _get_robot_from_env(self, env):
    if hasattr(env, "minitaur"):  #  Compabible with v1 envs.
      return env.minitaur
    elif hasattr(env, "robot"):  # Compatible with v2 envs.
      return env.robot
    else:
      return None

  def _build_randomization_function_dict(self, env):
    func_dict = {}
    robot = self._get_robot_from_env(env)
    func_dict["mass"] = functools.partial(
        self._randomize_masses, minitaur=robot)
    func_dict["inertia"] = functools.partial(
        self._randomize_inertia, minitaur=robot)
    func_dict["latency"] = functools.partial(
        self._randomize_latency, minitaur=robot)
    func_dict["joint friction"] = functools.partial(
        self._randomize_joint_friction, minitaur=robot)
    func_dict["motor friction"] = functools.partial(
        self._randomize_motor_friction, minitaur=robot)
    func_dict["restitution"] = functools.partial(
        self._randomize_contact_restitution, minitaur=robot)
    func_dict["lateral friction"] = functools.partial(
        self._randomize_contact_friction, minitaur=robot)
    func_dict["battery"] = functools.partial(
        self._randomize_battery_level, minitaur=robot)
    func_dict["motor strength"] = functools.partial(
        self._randomize_motor_strength, minitaur=robot)
    # Settinmg control step needs access to the environment.
    func_dict["control step"] = functools.partial(
        self._randomize_control_step, env=env)
    return func_dict

  def _randomize_control_step(self, env, lower_bound, upper_bound):
    randomized_control_step = random.uniform(lower_bound, upper_bound)
    env.set_time_step(randomized_control_step)
    tf.logging.info("control step is: {}".format(randomized_control_step))

  def _randomize_masses(self, minitaur, lower_bound, upper_bound):
    base_mass = minitaur.GetBaseMassesFromURDF()
    random_base_ratio = random.uniform(lower_bound, upper_bound)
    randomized_base_mass = random_base_ratio * np.array(base_mass)
    minitaur.SetBaseMasses(randomized_base_mass)
    tf.logging.info("base mass is: {}".format(randomized_base_mass))

    leg_masses = minitaur.GetLegMassesFromURDF()
    random_leg_ratio = random.uniform(lower_bound, upper_bound)
    randomized_leg_masses = random_leg_ratio * np.array(leg_masses)
    minitaur.SetLegMasses(randomized_leg_masses)
    tf.logging.info("leg mass is: {}".format(randomized_leg_masses))

  def _randomize_inertia(self, minitaur, lower_bound, upper_bound):
    base_inertia = minitaur.GetBaseInertiasFromURDF()
    random_base_ratio = random.uniform(lower_bound, upper_bound)
    randomized_base_inertia = random_base_ratio * np.array(base_inertia)
    minitaur.SetBaseInertias(randomized_base_inertia)
    tf.logging.info("base inertia is: {}".format(randomized_base_inertia))
    leg_inertia = minitaur.GetLegInertiasFromURDF()
    random_leg_ratio = random.uniform(lower_bound, upper_bound)
    randomized_leg_inertia = random_leg_ratio * np.array(leg_inertia)
    minitaur.SetLegInertias(randomized_leg_inertia)
    tf.logging.info("leg inertia is: {}".format(randomized_leg_inertia))

  def _randomize_latency(self, minitaur, lower_bound, upper_bound):
    randomized_latency = random.uniform(lower_bound, upper_bound)
    minitaur.SetControlLatency(randomized_latency)
    tf.logging.info("control latency is: {}".format(randomized_latency))

  def _randomize_joint_friction(self, minitaur, lower_bound, upper_bound):
    num_knee_joints = minitaur.GetNumKneeJoints()
    randomized_joint_frictions = np.random.uniform(
        [lower_bound] * num_knee_joints, [upper_bound] * num_knee_joints)
    minitaur.SetJointFriction(randomized_joint_frictions)
    tf.logging.info("joint friction is: {}".format(randomized_joint_frictions))

  def _randomize_motor_friction(self, minitaur, lower_bound, upper_bound):
    randomized_motor_damping = random.uniform(lower_bound, upper_bound)
    minitaur.SetMotorViscousDamping(randomized_motor_damping)
    tf.logging.info("motor friction is: {}".format(randomized_motor_damping))

  def _randomize_contact_restitution(self, minitaur, lower_bound, upper_bound):
    randomized_restitution = random.uniform(lower_bound, upper_bound)
    minitaur.SetFootRestitution(randomized_restitution)
    tf.logging.info("foot restitution is: {}".format(randomized_restitution))

  def _randomize_contact_friction(self, minitaur, lower_bound, upper_bound):
    randomized_foot_friction = random.uniform(lower_bound, upper_bound)
    minitaur.SetFootFriction(randomized_foot_friction)
    tf.logging.info("foot friction is: {}".format(randomized_foot_friction))

  def _randomize_battery_level(self, minitaur, lower_bound, upper_bound):
    randomized_battery_voltage = random.uniform(lower_bound, upper_bound)
    minitaur.SetBatteryVoltage(randomized_battery_voltage)
    tf.logging.info("battery voltage is: {}".format(randomized_battery_voltage))

  def _randomize_motor_strength(self, minitaur, lower_bound, upper_bound):
    randomized_motor_strength_ratios = np.random.uniform(
        [lower_bound] * minitaur.num_motors,
        [upper_bound] * minitaur.num_motors)
    minitaur.SetMotorStrengthRatios(randomized_motor_strength_ratios)
    tf.logging.info(
        "motor strength is: {}".format(randomized_motor_strength_ratios))
