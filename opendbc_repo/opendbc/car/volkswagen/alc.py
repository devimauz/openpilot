"""
Copyright © IQ.Lvbs, apart of Project Teal Lvbs, All Rights Reserved, licensed under https://konn3kt.com/tos
"""
import math
import os

from opendbc.car import DT_CTRL, structs
from opendbc.car.volkswagen import mqbcan, pqcan
from opendbc.car.volkswagen.values import CAR, CarControllerParams, VolkswagenFlags

IQLVBS_PQ_HCA_ADDR = 0x0D2
IQLVBS_PQ_APD_ADDR = 0x3D6
IQLVBS_MQB_APD_ADDR = 0x6A0
IQLVBS_ALC_STEER_ANGLE_MAX = 500.0
IQLVBS_PQ_HCA_MAX_RAW_ANGLE = 0x7FFF
IQLVBS_PQ_HCA_DEG_PER_BIT = 0.04375
IQLVBS_PQ_HCA_STATUS_READY = 3
IQLVBS_PQ_HCA_STATUS_ALC_ANGLE = 8
IQLVBS_PQ_HCA_STATUS_OVERRIDE_CLEAR = 9
IQLVBS_MQB_HCA_STATUS_READY = 3
IQLVBS_MQB_HCA_STATUS_ACTIVE = 7
IQLVBS_ALC_REENGAGE_WINDOW_FRAMES = int(10.0 / DT_CTRL)
IQLVBS_ALC_DOUBLE_SET_MAX_FRAMES = int(1.0 / DT_CTRL)
IQLVBS_TORQUE_ANGLE_SIGN_FLIP_MIN_ABS_ANGLE_DEG = 1.0
IQLVBS_TORQUE_ANGLE_ERROR_DEADBAND_DEG = 0.0
IQLVBS_TORQUE_ANGLE_KP_DEG_PER_ANGLE_ERR = 0.5
IQLVBS_TORQUE_ANGLE_MAX_ANGLE_ERR_BOOST_DEG = 3.0
IQLVBS_TORQUE_ANGLE_KP_DEG_PER_LAT_ACCEL = 5.0
IQLVBS_TORQUE_ANGLE_KD_DEG_PER_LAT_JERK = 0.015
IQLVBS_TORQUE_ANGLE_MAX_OVERSHOOT_DEG = 8.0
IQLVBS_TORQUE_ANGLE_OVERSHOOT_TAU_SEC = 0.0
IQLVBS_TORQUE_BLEND_DEADBAND_CNM = 150.0
IQLVBS_TORQUE_BLEND_PQ_SAT_CNM = 360.0
IQLVBS_TORQUE_BLEND_MQB_SAT_CNM = 550.0
IQLVBS_TORQUE_BLEND_PQ_MAX_OFFSET_DEG = 5.0
IQLVBS_TORQUE_BLEND_MQB_MAX_OFFSET_DEG = 7.0
IQLVBS_TORQUE_BLEND_TAU_SEC = 0.30
IQLVBS_MQB_PROGRESSIVE_ANGLE_MAP_X = [0.0, 50.0, 90.0, 130.0, 200.0, 360.0, 500.0]
IQLVBS_MQB_PROGRESSIVE_ANGLE_MAP_Y = [0.0, 50.0, 75.0, 110.0, 170.0, 320.0, 450.0]
IQLVBS_PQ_PASSAT_NMS_PROGRESSIVE_ANGLE_MAP_X = [0.0, 50.0, 56.25, 101.25, 146.25, 225.0, 405.0, 562.5]
IQLVBS_PQ_PASSAT_NMS_PROGRESSIVE_ANGLE_MAP_Y = [0.0, 50.0, 50.625, 84.375, 123.75, 191.25, 360.0, 506.25]
IQLVBS_APD_VERSION = 1
IQLVBS_APD_FLAG_CONFIG_VALID = 1 << 0
IQLVBS_APD_FLAG_ROLL_VALID = 1 << 1
IQLVBS_APD_FLAG_GRIP_VALID = 1 << 2
IQLVBS_APD_FLAG_AOL_ACTIVE = 1 << 3

ButtonType = structs.CarState.ButtonEvent.Type


def _iqlvbs_get_le_bits(data: bytes | bytearray, start_bit: int, bit_len: int) -> int:
  value = 0
  for i in range(bit_len):
    bit = start_bit + i
    value |= ((data[bit >> 3] >> (bit & 7)) & 0x1) << i
  return value


def _iqlvbs_set_le_bits(data: bytearray, start_bit: int, bit_len: int, value: int) -> None:
  for i in range(bit_len):
    bit = start_bit + i
    byte = bit >> 3
    mask = 1 << (bit & 7)
    if (value >> i) & 0x1:
      data[byte] |= mask
    else:
      data[byte] &= (~mask) & 0xFF


def _iqlvbs_pq_checksum(data: bytes | bytearray) -> int:
  return data[1] ^ data[2] ^ data[3] ^ data[4]


def _iqlvbs_get_param_int(params, key: str, default: int) -> int:
  if params is None:
    return default
  try:
    value = params.get(key, return_default=True)
    return default if value is None else int(value)
  except Exception:
    return default


def _iqlvbs_get_param_bool(params, key: str, default: bool = False) -> bool:
  if params is None:
    return default
  try:
    return bool(params.get_bool(key))
  except Exception:
    return default


def _iqlvbs_get_lat_smooth_sec(controller) -> float:
  return 0.0


def _iqlvbs_get_identity_prog_map(max_angle: float) -> tuple[list[float], list[float]]:
  max_angle = max(float(max_angle), 0.0)
  return [0.0, max_angle], [0.0, max_angle]


def _iqlvbs_get_pq_prog_map(controller) -> tuple[list[float], list[float]]:
  if getattr(controller.CP, "carFingerprint", None) == CAR.VOLKSWAGEN_PASSAT_NMS:
    return IQLVBS_PQ_PASSAT_NMS_PROGRESSIVE_ANGLE_MAP_X, IQLVBS_PQ_PASSAT_NMS_PROGRESSIVE_ANGLE_MAP_Y
  return _iqlvbs_get_identity_prog_map(IQLVBS_ALC_STEER_ANGLE_MAX)


def _iqlvbs_encode_apd_fields(steer_ratio: float, wheelbase: float, angle_offset_deg: float = 0.0,
                              grip_factor: float = 1.0, aol_active: bool = False) -> dict[str, int]:
  steer_ratio_raw = max(0, min(0x7FF, int(round((steer_ratio - 8.0) / 0.01))))
  wheelbase_mm = int(round(wheelbase * 1000.0))
  wheelbase_raw = max(0, min(0x7FF, wheelbase_mm - 2000))
  angle_offset_raw = max(0, min(0x7FF, int(round((angle_offset_deg + 10.0) / 0.01))))
  grip_factor_raw = max(0, min(0x7FF, int(round(grip_factor / 0.001))))
  return {
    "version": IQLVBS_APD_VERSION,
    "flags": IQLVBS_APD_FLAG_CONFIG_VALID | IQLVBS_APD_FLAG_ROLL_VALID | IQLVBS_APD_FLAG_GRIP_VALID |
             (IQLVBS_APD_FLAG_AOL_ACTIVE if aol_active else 0),
    "steer_ratio_raw": steer_ratio_raw,
    "wheelbase_raw": wheelbase_raw,
    "angle_offset_raw": angle_offset_raw,
    "grip_factor_raw": grip_factor_raw,
  }


def _iqlvbs_create_apd_1(packer, bus: int, steer_ratio: float, wheelbase: float, angle_offset_deg: float = 0.0,
                         grip_factor: float = 1.0, aol_active: bool = False):
  fields = _iqlvbs_encode_apd_fields(steer_ratio, wheelbase, angle_offset_deg, grip_factor, aol_active)
  values = {
    "APD_Version": fields["version"],
    "APD_Flags": fields["flags"],
    "APD_SteerRatio": fields["steer_ratio_raw"] * 0.01 + 8.0,
    "APD_Wheelbase": fields["wheelbase_raw"] + 2000,
    "APD_AngleOffSet": fields["angle_offset_raw"] * 0.01 - 10.0,
    "APD_GripFactor": fields["grip_factor_raw"] * 0.001,
  }
  return packer.make_can_msg("APD_1", bus, values)


def _iqlvbs_create_apd_raw(addr: int, bus: int, steer_ratio: float, wheelbase: float, angle_offset_deg: float = 0.0,
                           grip_factor: float = 1.0, aol_active: bool = False, counter: int = 0):
  fields = _iqlvbs_encode_apd_fields(steer_ratio, wheelbase, angle_offset_deg, grip_factor, aol_active)
  dat = bytearray(8)
  _iqlvbs_set_le_bits(dat, 8, 4, counter & 0x0F)
  _iqlvbs_set_le_bits(dat, 12, 4, fields["version"])
  _iqlvbs_set_le_bits(dat, 16, 4, fields["flags"])
  _iqlvbs_set_le_bits(dat, 20, 11, fields["steer_ratio_raw"])
  _iqlvbs_set_le_bits(dat, 31, 11, fields["wheelbase_raw"])
  _iqlvbs_set_le_bits(dat, 42, 11, fields["angle_offset_raw"])
  _iqlvbs_set_le_bits(dat, 53, 11, fields["grip_factor_raw"])
  dat[0] = 0
  for i in range(1, len(dat)):
    dat[0] ^= dat[i]
  return addr, bytes(dat), bus


def append_private_apd(controller, CC_IQ, can_sends) -> None:
  if controller.CP.flags & (VolkswagenFlags.MEB | VolkswagenFlags.MQB_EVO | VolkswagenFlags.MLB):
    return

  steer_ratio = float(getattr(controller.CP, "steerRatio", 0.0))
  wheelbase = float(getattr(controller.CP, "wheelbase", 0.0))
  angle_offset_deg = float(getattr(CC_IQ, "angleOffsetDeg", 0.0))
  grip_factor = 1.0
  aol_active = bool(getattr(getattr(CC_IQ, "aol", None), "active", False))

  if controller.CCS == pqcan:
    can_sends.append(_iqlvbs_create_apd_1(controller.packer_pt, 1, steer_ratio, wheelbase, angle_offset_deg, grip_factor, aol_active))
  elif controller.CCS == mqbcan:
    counter = int(getattr(controller, "_iqlvbs_mqb_apd_counter", 0)) & 0x0F
    can_sends.append(_iqlvbs_create_apd_raw(IQLVBS_MQB_APD_ADDR, 1, steer_ratio, wheelbase, angle_offset_deg, grip_factor, aol_active, counter))
    controller._iqlvbs_mqb_apd_counter = (counter + 1) & 0x0F


def _iqlvbs_get_message_data(pt_cp, msg_name: str) -> bytes:
  dat_map = getattr(pt_cp, "dat", {})
  if msg_name not in dat_map:
    try:
      _ = pt_cp.vl[msg_name]
    except Exception:
      pass
    dat_map = getattr(pt_cp, "dat", {})
  dat = dat_map.get(msg_name, b"")
  return dat if isinstance(dat, (bytes, bytearray)) else b""


def _iqlvbs_plain_alc_transport_marker(pt_cp) -> int:
  return 1


def _iqlvbs_decode_mqb_eps_pla_status(pt_cp) -> int:
  dat = _iqlvbs_get_message_data(pt_cp, "LH_EPS_03")
  return _iqlvbs_get_le_bits(dat, 50, 4) if len(dat) >= 8 else 0


def _iqlvbs_decode_mqb_eps_pla_abbruch(pt_cp) -> int:
  dat = _iqlvbs_get_message_data(pt_cp, "LH_EPS_03")
  return _iqlvbs_get_le_bits(dat, 56, 4) if len(dat) >= 8 else 0


def _iqlvbs_decode_pq_alc_status(pt_cp) -> int:
  dat = _iqlvbs_get_message_data(pt_cp, "Lenkhilfe_2")
  return _iqlvbs_get_le_bits(dat, 22, 2) if len(dat) >= 7 else 0


def _iqlvbs_decode_pq_alc_abbr(pt_cp) -> int:
  dat = _iqlvbs_get_message_data(pt_cp, "Lenkhilfe_2")
  return _iqlvbs_get_le_bits(dat, 24, 4) if len(dat) >= 7 else 0


def _iqlvbs_decode_pq_alc_lwreq(pt_cp) -> int:
  dat = _iqlvbs_get_message_data(pt_cp, "Lenkhilfe_2")
  return _iqlvbs_get_le_bits(dat, 40, 8) if len(dat) >= 7 else 0


def _iqlvbs_smooth_angle_request(desired_angle: float, last_angle: float, smooth_sec: float, steer_step: int) -> float:
  if smooth_sec <= 0:
    return desired_angle

  dt = DT_CTRL * steer_step
  alpha = 1 - math.exp(-dt / smooth_sec)
  return alpha * desired_angle + (1 - alpha) * last_angle


def _iqlvbs_interp(x: float, xp: list[float], fp: list[float]) -> float:
  if not xp or not fp or len(xp) != len(fp):
    return x
  if x <= xp[0]:
    return fp[0]
  if x >= xp[-1]:
    return fp[-1]

  for i in range(1, len(xp)):
    if x <= xp[i]:
      x0, x1 = xp[i - 1], xp[i]
      y0, y1 = fp[i - 1], fp[i]
      if x1 <= x0:
        return y1
      return y0 + ((x - x0) * (y1 - y0) / (x1 - x0))

  return fp[-1]


def _iqlvbs_apply_progressive_angle_map(angle_deg: float, prog_map_x: list[float], prog_map_y: list[float]) -> float:
  sign = -1.0 if angle_deg < 0.0 else 1.0
  mapped = _iqlvbs_interp(abs(float(angle_deg)), prog_map_x, prog_map_y)
  return sign * float(mapped)


def _iqlvbs_apply_torque_blend(controller, CS, desired_angle: float, active: bool, state_prefix: str,
                               sat_cnm: float, max_offset_deg: float) -> float:
  offset_attr = f"{state_prefix}_offset"
  params = getattr(controller, "_params", None)
  enabled = _iqlvbs_get_param_bool(params, "ALCTorqueBlend", False)

  if not enabled or not active:
    setattr(controller, offset_attr, 0.0)
    return desired_angle

  torque_cnm = float(getattr(CS.out, "steeringTorque", 0.0))
  torque_abs = abs(torque_cnm)
  target_offset = 0.0

  if torque_abs > IQLVBS_TORQUE_BLEND_DEADBAND_CNM:
    span = max(float(sat_cnm) - IQLVBS_TORQUE_BLEND_DEADBAND_CNM, 1.0)
    x = max(0.0, min((torque_abs - IQLVBS_TORQUE_BLEND_DEADBAND_CNM) / span, 1.0))
    eased = x * x * (3.0 - 2.0 * x)
    sign = -1.0 if torque_cnm < 0.0 else 1.0
    target_offset = sign * float(max_offset_deg) * eased

  last_offset = float(getattr(controller, offset_attr, 0.0))
  dt = DT_CTRL * max(int(getattr(controller.CCP, "STEER_STEP", 1)), 1)
  tau = float(IQLVBS_TORQUE_BLEND_TAU_SEC)
  alpha = 1.0 if tau <= 0.0 else 1.0 - math.exp(-dt / tau)
  offset = last_offset + alpha * (target_offset - last_offset)
  setattr(controller, offset_attr, offset)
  return desired_angle + offset


def _iqlvbs_torque_like_angle(desired_angle_deg: float, actual_angle_deg: float, v_ego: float, roll: float, VM, kp_deg_per_lat_accel: float, kd_deg_per_lat_jerk: float, max_overshoot_deg: float, lat_accel_err_last: float, dt: float) -> tuple[float, float, float]:
  v_ego_clip = max(v_ego, 1.0)

  desired_curv = VM.calc_curvature(math.radians(desired_angle_deg), v_ego_clip, roll)
  actual_curv = VM.calc_curvature(math.radians(actual_angle_deg), v_ego_clip, roll)

  desired_lat_accel = desired_curv * v_ego_clip ** 2
  actual_lat_accel = actual_curv * v_ego_clip ** 2

  lat_accel_err = desired_lat_accel - actual_lat_accel
  lat_jerk_err = (lat_accel_err - lat_accel_err_last) / dt if dt > 0 else 0.0

  overshoot = kp_deg_per_lat_accel * lat_accel_err + kd_deg_per_lat_jerk * lat_jerk_err
  overshoot = max(min(float(overshoot), max_overshoot_deg), -max_overshoot_deg)

  return desired_angle_deg + overshoot, lat_accel_err, overshoot


def _iqlvbs_apply_torque_like_angle(controller, CS, actual_angle_deg: float, safe_desired_angle_deg: float,
                                    active: bool, state_attr_prefix: str) -> float:
  err_attr = f"{state_attr_prefix}_lat_accel_err_last"
  overshoot_attr = f"{state_attr_prefix}_overshoot_last"
  active_last_attr = f"{state_attr_prefix}_active_last"

  if active and not bool(getattr(controller, active_last_attr, False)):
    setattr(controller, err_attr, 0.0)
    setattr(controller, overshoot_attr, 0.0)
  setattr(controller, active_last_attr, bool(active))

  if not active:
    setattr(controller, err_attr, 0.0)
    setattr(controller, overshoot_attr, 0.0)
    return safe_desired_angle_deg

  if abs(float(safe_desired_angle_deg) - float(actual_angle_deg)) <= IQLVBS_TORQUE_ANGLE_ERROR_DEADBAND_DEG:
    setattr(controller, err_attr, 0.0)
    setattr(controller, overshoot_attr, 0.0)
    return safe_desired_angle_deg

  if (abs(float(safe_desired_angle_deg)) >= IQLVBS_TORQUE_ANGLE_SIGN_FLIP_MIN_ABS_ANGLE_DEG and
      abs(float(actual_angle_deg)) >= IQLVBS_TORQUE_ANGLE_SIGN_FLIP_MIN_ABS_ANGLE_DEG and
      ((float(safe_desired_angle_deg) > 0.0 and float(actual_angle_deg) < 0.0) or
       (float(safe_desired_angle_deg) < 0.0 and float(actual_angle_deg) > 0.0))):
    setattr(controller, err_attr, 0.0)
    setattr(controller, overshoot_attr, 0.0)
    return safe_desired_angle_deg

  kp = IQLVBS_TORQUE_ANGLE_KP_DEG_PER_LAT_ACCEL
  kd = IQLVBS_TORQUE_ANGLE_KD_DEG_PER_LAT_JERK
  max_overshoot = IQLVBS_TORQUE_ANGLE_MAX_OVERSHOOT_DEG
  lat_accel_err_last = float(getattr(controller, err_attr, 0.0))

  _, lat_accel_err, overshoot = _iqlvbs_torque_like_angle(
    float(safe_desired_angle_deg),
    float(actual_angle_deg),
    float(CS.out.vEgoRaw),
    0.0,
    controller.VM,
    kp,
    kd,
    max_overshoot,
    lat_accel_err_last,
    DT_CTRL * controller.CCP.STEER_STEP,
  )

  setattr(controller, err_attr, float(lat_accel_err))
  angle_err = float(safe_desired_angle_deg) - float(actual_angle_deg)
  angle_err_boost = angle_err * IQLVBS_TORQUE_ANGLE_KP_DEG_PER_ANGLE_ERR
  angle_err_boost = max(min(angle_err_boost, IQLVBS_TORQUE_ANGLE_MAX_ANGLE_ERR_BOOST_DEG),
                        -IQLVBS_TORQUE_ANGLE_MAX_ANGLE_ERR_BOOST_DEG)
  overshoot = max(min(float(overshoot) + angle_err_boost, max_overshoot), -max_overshoot)

  overshoot_last = float(getattr(controller, overshoot_attr, 0.0))
  dt = DT_CTRL * max(int(getattr(controller.CCP, "STEER_STEP", 1)), 1)
  alpha = 1.0 if IQLVBS_TORQUE_ANGLE_OVERSHOOT_TAU_SEC <= 0.0 else 1.0 - math.exp(-dt / IQLVBS_TORQUE_ANGLE_OVERSHOOT_TAU_SEC)
  smoothed_overshoot = overshoot_last + alpha * (float(overshoot) - overshoot_last)
  commanded_angle = float(safe_desired_angle_deg) + smoothed_overshoot

  setattr(controller, overshoot_attr, float(smoothed_overshoot))
  return float(max(min(commanded_angle, IQLVBS_ALC_STEER_ANGLE_MAX), -IQLVBS_ALC_STEER_ANGLE_MAX))


def _iqlvbs_get_pre_torque_angle_last(controller, state_attr_prefix: str) -> float:
  return float(getattr(controller, f"{state_attr_prefix}_pre_torque_angle_last", float(getattr(controller, "apply_angle_last", 0.0))))


def _iqlvbs_set_pre_torque_angle_last(controller, state_attr_prefix: str, angle_deg: float) -> None:
  setattr(controller, f"{state_attr_prefix}_pre_torque_angle_last", float(angle_deg))


def _iqlvbs_apply_angle_limits(desired_angle: float, last_angle: float, active: bool) -> float:
  if not active:
    return last_angle
  return float(max(min(desired_angle, IQLVBS_ALC_STEER_ANGLE_MAX), -IQLVBS_ALC_STEER_ANGLE_MAX))


def _iqlvbs_make_pq_alc_frame(controller, bus: int, apply_angle: float, alc_status: int):
  counter_nibble = int(getattr(controller, "iqlvbs_pq_hca_counter", 0)) & 0xF
  angle_raw = min(int(round(abs(apply_angle) / IQLVBS_PQ_HCA_DEG_PER_BIT)), IQLVBS_PQ_HCA_MAX_RAW_ANGLE)
  data = bytearray(5)
  data[1] = ((alc_status & 0xF) << 4) | counter_nibble
  _iqlvbs_set_le_bits(data, 16, 15, angle_raw)
  _iqlvbs_set_le_bits(data, 31, 1, 1 if apply_angle < 0 else 0)
  data[0] = _iqlvbs_pq_checksum(data)
  controller.iqlvbs_pq_hca_counter = (counter_nibble + 1) & 0xF
  return IQLVBS_PQ_HCA_ADDR, bytes(data), bus


def _create_mqb_alc_steering_control(packer, bus, apply_angle, alc_status):
  values = {
    "HCA_01_LM_OffSign": 1 if apply_angle < 0 else 0,
    "HCA_01_Status_HCA": alc_status,
    "HCA_01_Vib_Freq": 16,
    "HCA_01_Sendestatus": 1 if alc_status == 7 else 0,
  }
  address, dat, bus = packer.make_can_msg("HCA_01", bus, values)
  data = bytearray(dat)
  angle_raw = min(int(round(abs(float(apply_angle)) / 0.1)), 0x1FFF)
  _iqlvbs_set_le_bits(data, 40, 13, angle_raw)
  _iqlvbs_set_le_bits(data, 31, 1, 1 if apply_angle < 0 else 0)
  return address, bytes(data), bus


def angle_lateral_control_requested(controller) -> bool:
  if controller.CCS not in (mqbcan, pqcan):
    return False

  return controller._params.get_bool("AngleLateralControl")


def angle_lateral_control_enabled(controller, CS) -> bool:
  return angle_lateral_control_requested(controller)


def angle_lateral_control_transport_enabled(controller, CS) -> bool:
  return angle_lateral_control_requested(controller)


def pq_exported_alc_available(status: int) -> bool:
  return int(status) in (1, 2, 3)


def pq_exported_alc_entering(status: int) -> bool:
  return int(status) == 1


def pq_exported_alc_active(status: int) -> bool:
  return int(status) == 2


def mqb_exported_alc_available(status: int) -> bool:
  return int(status) in (6, 8)


def mqb_exported_alc_entering(status: int) -> bool:
  return False


def mqb_exported_alc_active(status: int) -> bool:
  return int(status) == 6


def exported_alc_entering(CS) -> bool:
  return bool(getattr(CS, "vw_iq_lvbs_alc_entering", False))


def exported_alc_active(CS) -> bool:
  return bool(getattr(CS, "vw_iq_lvbs_alc_active", False))


def exported_alc_engaged(CS) -> bool:
  return exported_alc_active(CS) or exported_alc_entering(CS)


def update_mqb_carstate_alc_state(CS, pt_cp) -> None:
  eps_pla_status_raw = _iqlvbs_decode_mqb_eps_pla_status(pt_cp)
  eps_pla_abbruch_raw = _iqlvbs_decode_mqb_eps_pla_abbruch(pt_cp)
  CS.vw_iq_lvbs_alc_key_slot = _iqlvbs_plain_alc_transport_marker(pt_cp)
  CS.EPS_ALC_Status = 8 if eps_pla_status_raw == 11 else eps_pla_status_raw
  CS.vw_iq_lvbs_alc_available = mqb_exported_alc_available(CS.EPS_ALC_Status)
  CS.vw_iq_lvbs_alc_entering = mqb_exported_alc_entering(CS.EPS_ALC_Status)
  CS.vw_iq_lvbs_alc_active = mqb_exported_alc_active(CS.EPS_ALC_Status)
  CS.EPS_ALC_Abbruch = eps_pla_abbruch_raw


def update_pq_carstate_alc_state(CS, pt_cp) -> None:
  CS.PQ_ALC_Status_raw = _iqlvbs_decode_pq_alc_status(pt_cp)
  CS.LH2_Abbr = _iqlvbs_decode_pq_alc_abbr(pt_cp)
  CS.vw_iq_lvbs_alc_key_slot = _iqlvbs_plain_alc_transport_marker(pt_cp)
  CS.vw_iq_lvbs_alc_available = pq_exported_alc_available(CS.PQ_ALC_Status_raw)
  CS.vw_iq_lvbs_alc_entering = pq_exported_alc_entering(CS.PQ_ALC_Status_raw)
  CS.vw_iq_lvbs_alc_active = pq_exported_alc_active(CS.PQ_ALC_Status_raw)


def _iqlvbs_reset_reengage_state(controller) -> None:
  controller.ALC_override_counter = 0
  controller.ALC_reengage_set_tap_frames = 0
  controller.ALC_reengage_set_tap_count = 0


def _iqlvbs_button_events(CS) -> list:
  return list(getattr(getattr(CS, "out", None), "buttonEvents", []))


def _iqlvbs_set_released(controller, CS) -> bool:
  raw_set_pressed = None
  gra_stock_values = getattr(CS, "gra_stock_values", None)
  if isinstance(gra_stock_values, dict) and "GRA_Neu_Setzen" in gra_stock_values:
    raw_set_pressed = bool(gra_stock_values["GRA_Neu_Setzen"])

  last_raw = getattr(controller, "ALC_reengage_set_raw_last", None)
  raw_release = bool(last_raw) and (raw_set_pressed is False)
  controller.ALC_reengage_set_raw_last = raw_set_pressed

  button_release = any(be.type == ButtonType.setCruise and not be.pressed for be in _iqlvbs_button_events(CS))
  return raw_release or button_release


def _iqlvbs_double_set_reengage_requested(controller, CS, override_active: bool, *, allow_without_window: bool = False) -> bool:
  set_released = _iqlvbs_set_released(controller, CS)
  tap_frames = int(getattr(controller, "ALC_reengage_set_tap_frames", 0))
  tap_count = int(getattr(controller, "ALC_reengage_set_tap_count", 0))
  override_rising = override_active and not bool(getattr(controller, "ALC_override_last", False))
  controller.ALC_override_last = override_active

  if override_active:
    controller.ALC_override_counter = IQLVBS_ALC_REENGAGE_WINDOW_FRAMES
    if override_rising:
      tap_frames = 0
      tap_count = 0
    controller.ALC_reengage_set_tap_frames = tap_frames
    controller.ALC_reengage_set_tap_count = tap_count
    return False

  if int(getattr(controller, "ALC_override_counter", 0)) > 0:
    controller.ALC_override_counter -= 1
  elif not allow_without_window:
    _iqlvbs_reset_reengage_state(controller)
    return False

  if tap_frames > 0:
    tap_frames -= 1
    if tap_frames == 0:
      tap_count = 0

  if not set_released:
    controller.ALC_reengage_set_tap_frames = tap_frames
    controller.ALC_reengage_set_tap_count = tap_count
    return False

  if tap_count == 0:
    controller.ALC_reengage_set_tap_frames = IQLVBS_ALC_DOUBLE_SET_MAX_FRAMES
    controller.ALC_reengage_set_tap_count = 1
    return False

  _iqlvbs_reset_reengage_state(controller)
  return True


def _run_pq_alc(controller, CC, CS, actuators, can_sends) -> None:
  controller.ALC_feedback_entering = exported_alc_entering(CS)
  controller.ALC_feedback_active = exported_alc_active(CS)
  controller.ALC_feedback_engaged = exported_alc_engaged(CS)
  override_latched = bool(getattr(controller, "ALC_override_latched", False))
  override_clear_frames = int(getattr(controller, "ALC_override_clear_frames", 0))
  override_rearm_saw_inactive = bool(getattr(controller, "ALC_override_rearm_saw_inactive", False))
  cruise_main_enabled = bool(getattr(CS, "cruise_main_switch", True))
  lateral_available = bool(getattr(CS.out, "lateralAvailable", CS.out.cruiseState.available))

  if override_latched and not cruise_main_enabled:
    override_latched = False
    override_clear_frames = 0
    override_rearm_saw_inactive = False
    _iqlvbs_reset_reengage_state(controller)

  controller.ALC_driverExit = CS.LH2_Abbr in (2, 3, 4, 7) and lateral_available
  reengage_requested = _iqlvbs_double_set_reengage_requested(
    controller,
    CS,
    controller.ALC_driverExit,
    allow_without_window=override_latched,
  )

  if controller.ALC_driverExit:
    override_latched = True
    override_clear_frames = 2
    override_rearm_saw_inactive = False

  if override_latched and not CC.latActive:
    override_rearm_saw_inactive = True

  if override_latched and (override_clear_frames == 0) and ((override_rearm_saw_inactive and CC.latActive) or reengage_requested):
    override_latched = False
    override_rearm_saw_inactive = False

  controller.ALC_override_latched = override_latched
  controller.ALC_override_clear_frames = override_clear_frames
  controller.ALC_override_rearm_saw_inactive = override_rearm_saw_inactive
  CS.alcOverrideAlert = override_latched and not override_rearm_saw_inactive

  if override_latched:
    if override_clear_frames > 0:
      controller.ALC_status = IQLVBS_PQ_HCA_STATUS_OVERRIDE_CLEAR
      controller.ALC_override_clear_frames = override_clear_frames - 1
    else:
      controller.ALC_status = IQLVBS_PQ_HCA_STATUS_READY
  elif CC.latActive and not controller.ALC_driverExit:
    controller.ALC_status = IQLVBS_PQ_HCA_STATUS_ALC_ANGLE
    controller.ALC_entryCounter = 0
  else:
    controller.ALC_status = IQLVBS_PQ_HCA_STATUS_READY
    controller.ALC_entryCounter = 0
  pre_torque_angle_last = _iqlvbs_get_pre_torque_angle_last(controller, "pq_torque_like_angle")
  desired_angle = _iqlvbs_apply_progressive_angle_map(
    actuators.steeringAngleDeg,
    *_iqlvbs_get_pq_prog_map(controller),
  )
  desired_angle = _iqlvbs_apply_torque_blend(
    controller,
    CS,
    desired_angle,
    controller.ALC_status == IQLVBS_PQ_HCA_STATUS_ALC_ANGLE,
    "pq_torque_blend",
    IQLVBS_TORQUE_BLEND_PQ_SAT_CNM,
    IQLVBS_TORQUE_BLEND_PQ_MAX_OFFSET_DEG,
  )
  lat_smooth_sec = _iqlvbs_get_lat_smooth_sec(controller)
  desired_angle = _iqlvbs_smooth_angle_request(desired_angle, pre_torque_angle_last,
                                               lat_smooth_sec, controller.CCP.STEER_STEP)
  safe_desired_angle = _iqlvbs_apply_angle_limits(
    desired_angle,
    pre_torque_angle_last,
    controller.ALC_status == IQLVBS_PQ_HCA_STATUS_ALC_ANGLE,
  )
  apply_angle = _iqlvbs_apply_torque_like_angle(
    controller,
    CS,
    CS.out.steeringAngleDeg,
    safe_desired_angle,
    controller.ALC_status == IQLVBS_PQ_HCA_STATUS_ALC_ANGLE,
    "pq_torque_like_angle",
  )
  _iqlvbs_set_pre_torque_angle_last(controller, "pq_torque_like_angle", safe_desired_angle)

  if controller.ALC_feedback_engaged and controller.ALC_status == IQLVBS_PQ_HCA_STATUS_ALC_ANGLE:
    controller.alc_rack_angle = apply_angle

  controller.apply_angle_last = apply_angle
  controller.CSsteeringAngleDegLast = CS.out.steeringAngleDeg
  alc_msg = _iqlvbs_make_pq_alc_frame(controller, controller.CAN.cam, apply_angle, controller.ALC_status)
  can_sends.append(alc_msg)
  controller.CSLH3_SignLast = CS.LH_3_Sign


def _run_mqb_alc(controller, CC, CS, actuators, can_sends, apply_torque) -> None:
  controller.ALC_feedback_entering = exported_alc_entering(CS)
  controller.ALC_feedback_active = exported_alc_active(CS)
  controller.ALC_feedback_engaged = exported_alc_engaged(CS)
  controller.ALC_driverExit = (CS.EPS_ALC_Status == 10 or CS.EPS_ALC_Abbruch in (7, 10)) and CS.out.cruiseState.available
  reengage_requested = _iqlvbs_double_set_reengage_requested(controller, CS, controller.ALC_driverExit)

  if controller.ALC_driverExit:
    controller.ALC_reentry_blocked = True
  elif reengage_requested:
    controller.ALC_reentry_blocked = False
  elif not CC.latActive:
    controller.ALC_reentry_blocked = False

  if CC.latActive and not controller.ALC_driverExit and not controller.ALC_reentry_blocked:
    controller.ALC_status = IQLVBS_MQB_HCA_STATUS_ACTIVE
  else:
    controller.ALC_status = IQLVBS_MQB_HCA_STATUS_READY

  pre_torque_angle_last = _iqlvbs_get_pre_torque_angle_last(controller, "mqb_torque_like_angle")
  desired_angle = _iqlvbs_apply_progressive_angle_map(
    actuators.steeringAngleDeg,
    IQLVBS_MQB_PROGRESSIVE_ANGLE_MAP_X,
    IQLVBS_MQB_PROGRESSIVE_ANGLE_MAP_Y,
  )
  desired_angle = _iqlvbs_apply_torque_blend(
    controller,
    CS,
    desired_angle,
    controller.ALC_status == IQLVBS_MQB_HCA_STATUS_ACTIVE,
    "mqb_torque_blend",
    IQLVBS_TORQUE_BLEND_MQB_SAT_CNM,
    IQLVBS_TORQUE_BLEND_MQB_MAX_OFFSET_DEG,
  )
  lat_smooth_sec = _iqlvbs_get_lat_smooth_sec(controller)
  desired_angle = _iqlvbs_smooth_angle_request(desired_angle, pre_torque_angle_last,
                                               lat_smooth_sec, controller.CCP.STEER_STEP)
  safe_desired_angle = _iqlvbs_apply_angle_limits(
    desired_angle,
    pre_torque_angle_last,
    controller.ALC_status == IQLVBS_MQB_HCA_STATUS_ACTIVE,
  )
  apply_angle = _iqlvbs_apply_torque_like_angle(
    controller,
    CS,
    CS.out.steeringAngleDeg,
    safe_desired_angle,
    controller.ALC_status == IQLVBS_MQB_HCA_STATUS_ACTIVE,
    "mqb_torque_like_angle",
  )
  _iqlvbs_set_pre_torque_angle_last(controller, "mqb_torque_like_angle", safe_desired_angle)

  if controller.ALC_feedback_engaged and controller.ALC_status == IQLVBS_MQB_HCA_STATUS_ACTIVE:
    controller.alc_rack_angle = apply_angle

  controller.apply_angle_last = apply_angle
  controller.CSsteeringAngleDegLast = CS.out.steeringAngleDeg
  alc_msg = _create_mqb_alc_steering_control(
    controller.packer_pt, controller.CAN.pt, apply_angle, controller.ALC_status,
  )
  can_sends.append(alc_msg)

  if controller.CP.flags & VolkswagenFlags.STOCK_HCA_PRESENT:
    ea_simulated_torque = float(max(min(apply_torque * 2, controller.CCP.STEER_MAX), -controller.CCP.STEER_MAX))
    if abs(CS.out.steeringTorque) > abs(ea_simulated_torque):
      ea_simulated_torque = CS.out.steeringTorque
    can_sends.append(controller.CCS.create_eps_update(
      controller.packer_pt, controller.CAN.cam, CS.eps_stock_values, ea_simulated_torque,
    ))


_ALC_STATE_PATH = "/data/iqpilot_private/vw_alc_available"
_alc_available_last: bool | None = None


def _write_alc_availability(available: bool) -> None:
  global _alc_available_last
  if available == _alc_available_last:
    return
  try:
    os.makedirs(os.path.dirname(_ALC_STATE_PATH), exist_ok=True)
    with open(_ALC_STATE_PATH, "w") as f:
      f.write("1" if available else "0")
  except Exception:
    pass
  _alc_available_last = available


def update_vw_alc(controller, CC, CS, actuators, can_sends, apply_torque) -> None:
  CS.alcOverrideAlert = False
  controller.ALC_feedback_entering = exported_alc_entering(CS)
  controller.ALC_feedback_active = exported_alc_active(CS)
  controller.ALC_feedback_engaged = exported_alc_engaged(CS)

  _write_alc_availability(bool(getattr(CS, "vw_iq_lvbs_alc_available", False)))

  if controller.frame % controller.CCP.STEER_STEP != 0:
    return

  if not angle_lateral_control_enabled(controller, CS):
    return

  if controller.CCS == pqcan:
    _run_pq_alc(controller, CC, CS, actuators, can_sends)
  elif controller.CCS == mqbcan:
    _run_mqb_alc(controller, CC, CS, actuators, can_sends, apply_torque)
