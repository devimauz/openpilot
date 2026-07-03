#!/usr/bin/env python3
import numpy as np
from abc import ABC, abstractmethod

from openpilot.common.realtime import DT_HW
from openpilot.common.swaglog import cloudlog
from openpilot.common.pid import PIDController

class BaseFanController(ABC):
  @abstractmethod
  def update(self, cur_temp: float, ignition: bool) -> int:
    pass


class TiciFanController(BaseFanController):
  def __init__(self) -> None:
    super().__init__()
    cloudlog.info("Setting up TICI fan handler")

    self.last_ignition = False
    self.last_fan_pwr = 0
    self.target_temp_c = 75.0
    self.hyst_on_c = 1.5
    self.hyst_off_c = 0.8
    self.max_step_per_update = 2
    self.controller = PIDController(k_p=0, k_i=4e-3, k_f=1, rate=(1 / DT_HW))

  def update(self, cur_temp: float, ignition: bool) -> int:
    self.controller.neg_limit = -(100 if ignition else 30)
    self.controller.pos_limit = -(30 if ignition else 0)

    if ignition != self.last_ignition:
      self.controller.reset()

    error = self.target_temp_c - cur_temp
    fan_pwr_raw = -int(self.controller.update(
      error=error,
      feedforward=np.interp(cur_temp, [60.0, 100.0], [0, -100])
    ))

    max_pwr = 100 if ignition else 30
    fan_pwr_raw = int(np.clip(fan_pwr_raw, 0, max_pwr))

    # Temperature hysteresis prevents rapid 0<->nonzero toggling near the setpoint.
    if self.last_fan_pwr == 0 and cur_temp < (self.target_temp_c + self.hyst_on_c):
      fan_pwr_out = 0
    elif self.last_fan_pwr > 0 and cur_temp < (self.target_temp_c - self.hyst_off_c):
      fan_pwr_out = 0
    else:
      fan_pwr_out = fan_pwr_raw

    # Rate-limit command changes to avoid abrupt spin-up/spin-down oscillation.
    delta = fan_pwr_out - self.last_fan_pwr
    if delta > self.max_step_per_update:
      fan_pwr_out = self.last_fan_pwr + self.max_step_per_update
    elif delta < -self.max_step_per_update:
      fan_pwr_out = self.last_fan_pwr - self.max_step_per_update

    fan_pwr_out = int(np.clip(fan_pwr_out, 0, max_pwr))

    self.last_ignition = ignition
    self.last_fan_pwr = fan_pwr_out
    return fan_pwr_out

