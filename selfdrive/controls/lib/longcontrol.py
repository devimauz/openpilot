from cereal import car
from openpilot.common.numpy_fast import clip
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.selfdrive.controls.lib.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.common.params import Params

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

LongCtrlState = car.CarControl.Actuators.LongControlState


def long_control_state_trans(CP, active, long_control_state, v_ego,
                             should_stop, brake_pressed, cruise_standstill):
  stopping_condition = should_stop
  starting_condition = (not should_stop and
                        not cruise_standstill and
                        not brake_pressed)
  started_condition = v_ego > CP.vEgoStarting

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state in (LongCtrlState.off, LongCtrlState.pid):
      long_control_state = LongCtrlState.pid
      if stopping_condition: 
        #stoppingAccel = float(Params().get_int("StoppingAccel")) * 0.01    ### pid출력이 급정지(-accel) 상태에서 stopping으로 들어가면... 차량이 너무 급하게 섬.. 기다려보자.... 시험 230911
        #if a_target_now > stoppingAccel:  
        #  long_control_state = LongCtrlState.stopping
        long_control_state = LongCtrlState.stopping

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition and CP.startingState:
        long_control_state = LongCtrlState.starting
      elif starting_condition:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.starting:
      if stopping_condition:
        long_control_state = LongCtrlState.stopping
      elif started_condition:
        long_control_state = LongCtrlState.pid

  return long_control_state

class LongControl:
  def __init__(self, CP):
    self.CP = CP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController((CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV),
                             (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             k_f=CP.longitudinalTuning.kf, rate=1 / DT_CTRL)
    self.last_output_accel = 0.0
    self.readParamCount = 0
    self.longitudinalTuningKpV = 1.0
    self.longitudinalTuningKiV = 0.0
    self.longitudinalTuningKf = 1.0
    self.startAccelApply = 0.0
    self.stopAccelApply = 0.0

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, a_target, should_stop, accel_limits, softHoldActive):
  
    self.readParamCount += 1
    if self.readParamCount >= 100:
      self.readParamCount = 0
    elif self.readParamCount == 10:
      self.longitudinalTuningKpV = float(Params().get_int("LongitudinalTuningKpV")) * 0.01
      self.longitudinalTuningKiV = float(Params().get_int("LongitudinalTuningKiV")) * 0.001
      self.longitudinalTuningKf = float(Params().get_int("LongitudinalTuningKf")) * 0.01

      ## longcontrolTuning이 한개일때만 적용
      #if len(self.CP.longitudinalTuning.kpBP) == 1 and len(self.CP.longitudinalTuning.kiBP)==1:
      #  self.CP.longitudinalTuning.kpV = [self.longitudinalTuningKpV]
      #  self.CP.longitudinalTuning.kiV = [self.longitudinalTuningKiV]
      #  self.pid._k_p = (self.CP.longitudinalTuning.kpBP, self.CP.longitudinalTuning.kpV)
      #  self.pid._k_i = (self.CP.longitudinalTuning.kiBP, self.CP.longitudinalTuning.kiV)
      #  self.pid.k_f = self.longitudinalTuningKf
      #  #self.pid._k_i = ([0, 2.0, 200], [self.longitudinalTuningKiV, 0.0, 0.0]) # 정지때만.... i를 적용해보자... 시험..
    elif self.readParamCount == 30:
      pass
    elif self.readParamCount == 40:
      self.startAccelApply = float(Params().get_int("StartAccelApply")) * 0.01
      self.stopAccelApply = float(Params().get_int("StopAccelApply")) * 0.01
      
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    self.long_control_state = long_control_state_trans(self.CP, active, self.long_control_state, CS.vEgo,
                                                       should_stop, CS.brakePressed,
                                                       CS.cruiseState.standstill)

    if active and softHoldActive > 0:
      self.long_control_state = LongCtrlState.stopping

    if self.long_control_state == LongCtrlState.off:
      self.reset()
      output_accel = 0.

    elif self.long_control_state == LongCtrlState.stopping:
      output_accel = self.last_output_accel
      if output_accel > self.CP.stopAccel:
        output_accel = min(output_accel, 0.0)
        output_accel -= self.CP.stoppingDecelRate * DT_CTRL
        if softHoldActive > 0:
          output_accel = self.CP.stopAccel
      self.reset()

    elif self.long_control_state == LongCtrlState.starting:
      output_accel = self.CP.startAccel
      self.reset()

    else:  # LongCtrlState.pid
      error = a_target - CS.aEgo
      output_accel = self.pid.update(error, speed=CS.vEgo,
                                     feedforward=a_target)

    self.last_output_accel = clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel
