import numpy as np


def _apply_progressive_map(angle: float, prog_map_x, prog_map_y) -> float:
  if prog_map_x is None or prog_map_y is None or len(prog_map_x) != len(prog_map_y) or len(prog_map_x) < 2:
    return angle
  sign = -1.0 if angle < 0.0 else 1.0
  return sign * float(np.interp(abs(float(angle)), prog_map_x, prog_map_y))


def apply_steer_angle_limits_simple(apply_angle: float, apply_angle_last: float, v_ego_raw: float,
                                    steering_angle: float, lat_active: bool, CCP, VM,
                                    prog_map_x=None, prog_map_y=None) -> float:
  """Rate/abs-limit a commanded steering-wheel angle.

  Mirrors ``apply_std_steer_angle_limits`` but takes the ``CarControllerParams`` class
  (using ``CCP.ANGLE_LIMITS``) and accepts an optional progressive angle map. ``VM`` is
  accepted for signature compatibility with callers that pass a ``VehicleModel``.
  """
  limits = CCP.ANGLE_LIMITS

  apply_angle = _apply_progressive_map(apply_angle, prog_map_x, prog_map_y)

  # pick angle rate limits based on wind up/down
  steer_up = apply_angle_last * apply_angle >= 0. and abs(apply_angle) > abs(apply_angle_last)
  rate_limits = limits.ANGLE_RATE_LIMIT_UP if steer_up else limits.ANGLE_RATE_LIMIT_DOWN

  angle_rate_lim = np.interp(v_ego_raw, rate_limits[0], rate_limits[1])
  new_apply_angle = np.clip(apply_angle, apply_angle_last - angle_rate_lim, apply_angle_last + angle_rate_lim)

  # angle is current steering wheel angle when inactive on all angle cars
  if not lat_active:
    new_apply_angle = steering_angle

  return float(np.clip(new_apply_angle, -limits.STEER_ANGLE_MAX, limits.STEER_ANGLE_MAX))
