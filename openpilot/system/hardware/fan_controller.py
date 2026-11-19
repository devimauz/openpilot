#!/usr/bin/env python3
import numpy as np

from openpilot.common.pid import PIDController

C3_FAN_DEVICE_TYPES = frozenset(("tici", "tizi"))


class FanController:
  """C3/C3X: hysteresis profile (silent below 80C, 20% at 80C, ramps up, no I-term hunting).
  Other devices keep the stock PID controller."""

  # profile thresholds (deg C) -> fan percent
  C3_ON_TEMP = 80.0          # fan starts here
  C3_OFF_TEMP = 77.0         # fan stops only after cooling below this (deadband)
  C3_MIN_RUN_S = 90.0        # stay running this long once started (anti short-cycle)
  C3_RAMP_TEMPS = [80.0, 88.0, 93.0, 97.0]
  C3_RAMP_PCTS = [20.0, 45.0, 70.0, 100.0]
  C3_OFFROAD_MAX = 30.0

  def __init__(self, rate: int, device_type: str) -> None:
    self.rate = max(1, int(rate))
    self.c3_fan_profile = device_type in C3_FAN_DEVICE_TYPES
    self.last_ignition = False
    if self.c3_fan_profile:
      self.running = False
      self.run_ticks = 0
    else:
      self.target_temp = 70
      self.controller = PIDController(k_p=0, k_i=4e-3, k_f=0, rate=rate)

  def update(self, cur_temp: float, ignition: bool) -> int:
    if not self.c3_fan_profile:
      self.controller.pos_limit = 100 if ignition else 30
      self.controller.neg_limit = 30 if ignition else 0
      if ignition != self.last_ignition:
        self.controller.reset()
      self.last_ignition = ignition
      return int(self.controller.update(error=(cur_temp - self.target_temp),
                                        feedforward=np.interp(cur_temp, [60.0, 100.0], [0, 100])))

    if ignition != self.last_ignition:
      # reset hysteresis memory on ignition transitions, same as old controller did
      self.running = False
      self.run_ticks = 0
      self.last_ignition = ignition

    duty = 0.0
    if not self.running and cur_temp >= self.C3_ON_TEMP:
      self.running = True
      self.run_ticks = 0
    if self.running:
      self.run_ticks += 1
      duty = float(np.interp(cur_temp, self.C3_RAMP_TEMPS, self.C3_RAMP_PCTS))
      if not ignition:
        duty = min(duty, self.C3_OFFROAD_MAX)
      stop_temp = self.C3_OFF_TEMP if ignition else self.C3_OFF_TEMP - 2.0
      if cur_temp <= stop_temp and self.run_ticks >= self.C3_MIN_RUN_S * self.rate:
        self.running = False
        self.run_ticks = 0
        duty = 0.0
    return int(duty)
