"""TomTom guidance state -> carrot_navi v2 item values (pure functions, no I/O)."""
from __future__ import annotations

from dataclasses import dataclass, field

PROTOCOL_VERSION = 2
CATALOG_REVISION = 1
SCHEMA_VERSION = 1

JSON_NAMES = (
  "vehicle", "guidance_current", "guidance_next", "lane_current",
  "lane_ahead", "speed", "traffic_signal", "crossroad", "route",
  "navigation_status", "app_status", "camera_state", "composition_state",
)
IMAGE_NAMES = (
  "tbt_current_compact", "tbt_current_full", "tbt_next",
  "traffic_signal", "lane_top", "lane_bottom",
  "safety_primary", "safety_secondary", "safety_section",
  "crossroad_minimized", "crossroad_expanded",
  "center_tbt_icon", "center_tbt_text", "center_tbt_fee",
)
RENDER_NAMES = ("map_main",)
CATALOG = ([("json", n) for n in JSON_NAMES]
           + [("image", n) for n in IMAGE_NAMES]
           + [("render", n) for n in RENDER_NAMES])

# carrot nav_type_mapping codes (see README for table)
TURN = {"TURN_LEFT": 12, "TURN_RIGHT": 13, "SHARP_LEFT": 16, "SHARP_RIGHT": 19,
        "SLIGHT_LEFT": 1000, "SLIGHT_RIGHT": 1001, "MAKE_UTURN": 14,
        "ARRIVE": 201, "KEEP_LEFT": 44, "KEEP_RIGHT": 43,
        "ROUNDABOUT_STRAIGHT": 142, "ROUNDABOUT_SLIGHT_RIGHT": 131,
        "ROUNDABOUT_RIGHT": 133, "ROUNDABOUT_SHARP_RIGHT": 134,
        "ROUNDABOUT_SLIGHT_LEFT": 140, "ROUNDABOUT_LEFT": 139,
        "ROUNDABOUT_SHARP_LEFT": 136, "ROUNDABOUT_BACK": 135}

# fallback entry speeds (kph) when no curve radius is available
ENTRY_SPEED = {12: 30, 13: 30, 16: 20, 19: 20, 1000: 35, 1001: 35,
               6: 40, 7: 40, 1002: 40, 1003: 40, 131: 28, 133: 25, 142: 30,
               139: 28, 140: 30, 101: 40, 102: 40, 104: 40, 105: 40, 106: 40, 107: 40}

SPEED_CTRL_TYPES = (0, 1, 2, 3, 4, 7, 8, 75, 76)   # device-accepted sdi.type values
SDI_LIMIT_TYPE = 3
BLOCK_CURVE_TYPE = 2
ROAD_LIMIT_STEP = 10
MAX_ROUTE_POINTS = 256
NO_MANEUVER = {"DEPART", "WAYPOINT_LEFT", "WAYPOINT_RIGHT", "WAYPOINT_REACHED", "WAYPOINT_AHEAD"}


def tomtom_turn_type(maneuver: str) -> int | None:
  return TURN.get(maneuver.upper().replace("TURN_", "TURN_"))


def roundabout_exit_code(exit_number: int, num_exits: int) -> int:
  """Fallback when the SDK gives only exit numbers: map exit position to the
  rotary code by the turn angle it represents (right-Hand traffic circles)."""
  if num_exits <= 1:
    return 142
  frac = (exit_number - 1) / max(1, num_exits - 1)          # 0..1 around the circle
  if frac <= 0.15:
    return 131
  if frac <= 0.40:
    return 133
  if frac <= 0.55:
    return 142
  if frac <= 0.80:
    return 139
  return 136


def snap_road_limit(kph: int) -> int:
  v = int(kph) // ROAD_LIMIT_STEP * ROAD_LIMIT_STEP
  return v if 0 < v <= 200 else 0


@dataclass
class Maneuver:
  type: str                       # TomTom maneuver string
  distance_m: int = 0
  time_sec: int = 0
  road_name: str = ""
  main_text: str = ""
  point: tuple[float, float] | None = None      # (lat, lon)
  exit_number: int | None = None
  num_exits: int | None = None
  radius_m: float | None = None                 # curve radius if known

  def turn_code(self) -> int | None:
    code = tomtom_turn_type(self.type)
    if code is None and self.type.upper().startswith("EXIT_ROUNDABOUT") and self.exit_number:
      code = roundabout_exit_code(self.exit_number, self.num_exits or 4)
    return code

  def entry_speed_kph(self) -> int:
    if self.radius_m:
      v = (2.8 * self.radius_m) ** 0.5 * 3.6
      return max(15, min(60, int(v)))
    code = self.turn_code() or 0
    return ENTRY_SPEED.get(code, 0)


@dataclass
class LimitChange:
  distance_m: int
  limit_kph: int


@dataclass
class NaviSnapshot:
  lat: float
  lon: float
  heading_deg: float
  speed_kph: float
  current_limit_kph: int                       # now-valid limit (phone's map)
  next_change: LimitChange | None              # upcoming SPEED_LIMIT section start
  second_change: LimitChange | None
  maneuver_now: Maneuver | None                # active TBT (incl. rotary traversal)
  maneuver_next: Maneuver | None
  route_polyline: list[tuple[float, float]] = field(default_factory=list)
  route_remain_m: int = 0
  route_remain_sec: int = 0
  route_moved_m: int = 0
  route_total_m: int = 0
  guidance_active: bool = True
  off_route: bool = False
  road_name: str = ""

  def _guidance(self, m: Maneuver | None) -> dict:
    if m is None or m.type.upper() in NO_MANEUVER:
      return {"present": False, "reason": "source_absent"}
    code = m.turn_code()
    if code is None:
      return {"present": False, "reason": "invalid"}
    value = {"distance_m": max(0, int(m.distance_m)), "time_sec": max(0, int(m.time_sec)),
             "turn_type": code, "road_name": m.road_name[:96], "main_text": m.main_text[:160],
             "near_direction": "", "mid_direction": "", "far_direction": ""}
    if m.point:
      value["point"] = {"lat": m.point[0], "lon": m.point[1]}
    return {"present": True, "value": value}

  def _speed(self) -> dict:
    speed: dict = {"current_kph": round(self.speed_kph, 1)}
    limit = snap_road_limit(self.current_limit_kph)
    if limit:
      speed["road_limit_kph"] = limit

    candidates: list[tuple[int, dict, bool]] = []
    nc = self.next_change
    if nc and nc.limit_kph:
      candidates.append((int(nc.distance_m), {
        "type": SDI_LIMIT_TYPE, "distance_m": int(nc.distance_m),
        "speed_limit_kph": int(nc.limit_kph), "section_type": -1,
        "block_type": BLOCK_CURVE_TYPE,
        "block_speed_kph": int(snap_road_limit(nc.limit_kph) or nc.limit_kph),
        "block_distance_m": int(nc.distance_m)}, False))
    m = self.maneuver_now
    if m and m.type.upper().startswith(("ROUNDABOUT", "EXIT_ROUNDABOUT", "TURN_", "SHARP_",
                                         "SLIGHT_", "OFF_RAMP", "FORK")) and m.distance_m <= 350:
      entry = m.entry_speed_kph()
      if entry:
        candidates.append((int(m.distance_m), {
          "type": 3, "distance_m": int(m.distance_m), "speed_limit_kph": int(entry),
          "section_type": -1, "block_type": BLOCK_CURVE_TYPE,
          "block_speed_kph": int(entry), "block_distance_m": int(m.distance_m)}, True))
    demoted = None
    if candidates:
      # sdi slot priority: an active maneuver decel block (safety-critical) beats a
      # limit change; the loser is demoted to sdi_secondary so no info is dropped.
      limit_cand = next((c for c in candidates if not c[2]), None)
      man_cand = next((c for c in candidates if c[2]), None)
      if man_cand:
        speed["sdi"] = man_cand[1]
        demoted = limit_cand
      elif limit_cand:
        speed["sdi"] = limit_cand[1]
        demoted = None
      else:
        demoted = None
    sc = self.second_change
    secondary = None
    if demoted:
      demoted_val = dict(demoted[1])
      if sc is None or sc.distance_m is None or demoted[0] <= sc.distance_m:
        secondary = demoted_val
    if secondary is None and sc:
      secondary = {"type": SDI_LIMIT_TYPE, "distance_m": int(sc.distance_m),
                   "speed_limit_kph": int(sc.limit_kph), "section_type": -1,
                   "block_type": -1, "block_speed_kph": 0, "block_distance_m": 0}
    if secondary:
      speed["sdi_secondary"] = secondary
    return speed

  def build_items(self) -> dict[str, dict]:
    items = {
      "vehicle": {"present": True, "value": {
        "lat": self.lat, "lon": self.lon, "heading_deg": self.heading_deg,
        "speed_kph": self.speed_kph, "road_name": self.road_name[:96], "virtual_gps": True}},
      "guidance_current": self._guidance(self.maneuver_now),
      "guidance_next": self._guidance(self.maneuver_next),
      "speed": {"present": True, "value": self._speed()},
      "route": {"present": True, "value": {
        "remain_distance_m": int(self.route_remain_m), "remain_time_sec": int(self.route_remain_sec),
        "moved_distance_m": int(self.route_moved_m), "total_distance_m": int(self.route_total_m),
        "polyline": [{"lat": p[0], "lon": p[1]}
                     for p in self.route_polyline[:MAX_ROUTE_POINTS]]}},
      "navigation_status": {"present": True, "value": {
        "mode": "tomtom", "guidance_active": self.guidance_active,
        "off_route": self.off_route, "route_present": bool(self.route_polyline)}},
      "app_status": {"present": True, "value": {"foreground": True}},
    }
    for name in ("lane_current", "lane_ahead", "traffic_signal", "crossroad",
                 "camera_state", "composition_state"):
      items[name] = {"present": False, "reason": "source_absent"}
    items["lane_ahead"]["value"] = None
    return items


def requirements_query() -> dict:
  return {"type": "requirements_query", "protocol_version": PROTOCOL_VERSION,
          "catalog_revision": CATALOG_REVISION,
          "streams": [{"kind": k, "name": n, "schema_version": SCHEMA_VERSION}
                      for k, n in CATALOG]}
