"""Scenario simulator + offline self-validation for the carrot_navi v2 feed.

Timeline (1 Hz), a Szombathely-ish drive:
  t 0-30   cruise 50 km/h; NEXT LIMIT CHANGE 30 km/h approaching 400m -> 0m
  t 31-55  inside 30 zone; ROUNDABOUT right (exit 2 of 4) 300m -> 0, curve radius ~18m
  t 56-80  JUNCTION turn right 200m -> 0; limit change back to 50 at 150m
  t 81-90  ARRIVE
"""
from __future__ import annotations

import argparse
import json
import math
import time

from tomtom2carrot import (CATALOG, JSON_NAMES, LimitChange, Maneuver, NaviSnapshot,
                           PROTOCOL_VERSION, ROAD_LIMIT_STEP, SPEED_CTRL_TYPES,
                           requirements_query)

LAT0, LON0 = 47.2254, 16.6288          # Szombathely


def _polyline(n: int = 64) -> list[tuple[float, float]]:
  return [(LAT0 + 0.004 * math.sin(i / n * 2 * math.pi),
           LON0 + 0.005 * math.cos(i / n * 2 * math.pi)) for i in range(n)]


def scenario_snapshot(t: float) -> NaviSnapshot:
  s = NaviSnapshot(lat=LAT0 + 0.00001 * t, lon=LON0 + 0.000012 * t,
                   heading_deg=(90 + 2 * t) % 360, speed_kph=50 if t < 30 else 30,
                   current_limit_kph=50 if t < 31 else 30,
                   next_change=None, second_change=LimitChange(900, 50),
                   maneuver_now=None, maneuver_next=None,
                   route_polyline=_polyline(), route_remain_m=max(0, 4200 - int(t * 12)),
                   route_remain_sec=max(0, 600 - int(t * 2)), route_moved_m=int(t * 12),
                   route_total_m=4200, road_name="Köszeg út")
  if t <= 30:
    s.next_change = LimitChange(max(0, 400 - int(t * 13)), 30)
    s.maneuver_next = Maneuver("ROUNDABOUT_RIGHT", distance_m=max(0, 620 - int(t * 16)),
                               road_name="Fő tér", main_text="Roundabout - take exit 2")
  elif 31 <= t <= 55:
    s.current_limit_kph, s.next_change, s.second_change = 30, None, LimitChange(450, 50)
    d = max(0, 300 - int((t - 31) * 11))
    s.maneuver_now = Maneuver("EXIT_ROUNDABOUT", distance_m=d, road_name="Vörösmarty utca",
                              main_text="Roundabout - exit 2", exit_number=2, num_exits=4,
                              radius_m=18.0 if d < 120 else None,
                              point=(LAT0 + 0.0005, LON0 + 0.0006))
  elif 56 <= t <= 80:
    s.current_limit_kph = 30
    s.next_change = LimitChange(max(0, 150 - int((t - 56) * 6)), 50)
    d = max(0, 200 - int((t - 56) * 8))
    s.maneuver_now = Maneuver("TURN_RIGHT", distance_m=d, road_name="Széll Kálmán körút",
                              main_text="Right onto Tópark utca",
                              radius_m=14.0 if d < 80 else None, point=(LAT0 + 0.0009, LON0 + 0.0011))
  else:
    s.current_limit_kph, s.speed_kph = 50, 40
    s.maneuver_now = Maneuver("ARRIVE", distance_m=max(0, 120 - int((t - 81) * 5)))
  return s


# ---------------------------------------------------------------- selftest
def selftest() -> None:
  assert len(CATALOG) == 28, "receiver demands exactly 28 catalog streams"
  assert len(JSON_NAMES) == 13
  req = requirements_query()
  assert req["protocol_version"] == PROTOCOL_VERSION and req["catalog_revision"] == 1

  seen_seq: dict[str, int] = {}
  for t in [0, 5, 15, 29, 31, 40, 55, 60, 79, 81, 90]:
    snap = scenario_snapshot(t)
    items = snap.build_items()
    assert set(items) == set(JSON_NAMES), f"t={t}: item set mismatch"
    for name, item in items.items():
      if item.get("present"):
        v = item["value"]
        assert isinstance(v, dict) or (name == "lane_ahead" and isinstance(v, list)), name
        if name == "speed":
          lim = v.get("road_limit_kph", 0)
          assert lim == 0 or (0 < lim <= 200 and lim % ROAD_LIMIT_STEP == 0), f"t={t} limit {lim}"
          sdi = v.get("sdi")
          if sdi:
            assert sdi["type"] in SPEED_CTRL_TYPES, sdi
            assert 0 <= sdi["distance_m"] <= 2_000_000
            assert 0 < sdi["speed_limit_kph"] <= 300
            if sdi["block_type"] == 2:
              assert 0 < sdi["block_speed_kph"] <= 300 and sdi["block_distance_m"] >= 0
        if name in ("guidance_current", "guidance_next"):
          assert -1 <= v["turn_type"] <= 100_000 and v["distance_m"] >= 0, v
      else:
        assert "reason" in item, name
        if name in ("vehicle", "speed", "route"):
          raise AssertionError(f"core item {name} must be present")
    # envelope shape (mirrors NaviFeedClient._envelope, checked against receiver rules)
    for name in ("vehicle", "speed", "guidance_current"):
      item = items[name]
      env = {"type": "item_update", "protocol_version": PROTOCOL_VERSION, "session_id": "x" * 16,
             "kind": "json", "name": name, "sequence": seen_seq.get(name, 0) + 1,
             "source_timestamp_ms": int(time.monotonic() * 1000), "sent_at_ms": int(time.monotonic() * 1000),
             "present": item["present"], "value": item.get("value")}
      if not item["present"]:
        env["value"], env["reason"] = None, item["reason"]
      assert env["sequence"] > seen_seq.get(name, 0)
      seen_seq[name] = env["sequence"]
      assert ("value" in env) and (env["present"] or env["value"] is None)

  # spot-check semantic expectations
  m = scenario_snapshot(45).maneuver_now
  assert m.turn_code() == 133, f"roundabout right exit -> 133, got {m.turn_code()}"
  assert scenario_snapshot(65).maneuver_now.turn_code() == 13
  sp = scenario_snapshot(20).build_items()["speed"]["value"]
  assert sp["sdi"]["speed_limit_kph"] == 30 and sp["road_limit_kph"] == 50
  sp45 = scenario_snapshot(45).build_items()["speed"]["value"]
  assert sp45["sdi"]["distance_m"] == 146 and sp45["sdi"]["speed_limit_kph"] <= 30, \
    f"roundabout decel block must ride in sdi without any limit change: {sp45.get('sdi')}"
  sp60 = scenario_snapshot(60).build_items()["speed"]["value"]
  assert sp60["road_limit_kph"] == 30 and sp60["sdi"]["speed_limit_kph"] == 30 \
    and sp60["sdi"]["distance_m"] == 168, f"turn block must beat limit lift: {sp60['sdi']}"
  sp65 = scenario_snapshot(65).build_items()["speed"]["value"]
  assert sp65["sdi"]["block_speed_kph"] == 30 and sp65["sdi_secondary"]["speed_limit_kph"] == 50, \
    f"limit lift demoted to secondary: {sp65}"
  print("SELFTEST PASS - 28 catalog, envelopes, limits (10-step), sdi types, " +
        "turn codes (rotary 133 / turn 13), distance+limit scenarios OK")


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--selftest", action="store_true")
  ap.add_argument("--dry", action="store_true")
  ap.add_argument("--seconds", type=int, default=90)
  ap.add_argument("--host")
  args = ap.parse_args()

  if args.selftest:
    selftest()
    return
  if args.host:
    import asyncio
    from feed_client import probe
    asyncio.run(probe(args.host, ticks=args.seconds // 3, period=1.0))
    return

  frames = 0
  for t in range(args.seconds):
    items = scenario_snapshot(t).build_items()
    watch = {n: items[n] for n in ("speed", "guidance_current")}
    line = json.dumps(watch, ensure_ascii=False)
    print(f"t={t:3d}s {line[:400]}")
    frames += 1
  print(f"(dry run: {frames} ticks; drop --dry into --host <comma-ip> when the device is up)")


if __name__ == "__main__":
  main()
