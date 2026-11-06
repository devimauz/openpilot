# TomTom phone → comma (carrot_navi) feed — spec

Phone runs TomTom SDK (Android) and streams navigation state to the comma device using
carrot's **external navigator v2 protocol** (the one built for Korean TMAP). On-device this
feeds `carrotNavi` cereal → `carrot_serv`/`CarrotNaviControl` → speed-limit decel
(`AutoNaviSpeedCtrlMode`), turn/roundabout maneuver display, cluster icons, and AutoTurn.

## Transport
- **Control ws** : `ws://<comma>:7714/api/navi/ws/v2/control/<app-version-string>` (text JSON).
  One owner at a time — second connect gets `receiver_busy`, so close your control ws on exit.
- **Item ws** (one per stream): `ws://<comma>:7714/api/navi/ws/v2/json/{session_id}/{name}`.
- **Discovery**: comma broadcasts UDP on **:7705** every 1 s — phone may listen to auto-find the IP;
  otherwise static host (home Wi-Fi: current LAN IP; on road: comma's IP from the phone's hotspot).
- Image/render streams exist in the catalog but we never open them; only `json` items are sent.

## Handshake
1. Phone sends on control ws:
```json
{"type":"requirements_query","protocol_version":2,"catalog_revision":1,
 "streams":[{"kind":"json","name":"vehicle","schema_version":1}, … 28 entries total …]}
```
   Catalog MUST equal exactly 13 json + 14 image + 1 render names (see `CATALOG` in feed_client.py).
2. Device replies `{"type":"subscription_manifest","session_id":"<hex>", "streams":[...]}`.
3. Phone opens a json item ws per name using that session_id and pushes envelopes.

## Item update envelope (one per item, on its own ws)
```json
{"type":"item_update","protocol_version":2,"session_id":"<sid>","kind":"json","name":"speed",
 "sequence":42,"source_timestamp_ms":1793940000000,"sent_at_ms":1793940000000,
 "present":true,"value":{...}}
```
- `sequence` strictly increases per stream. `present:false` ⇒ `value:null` +
  `reason ∈ source_absent|cleared|expired|passed|invalid`.
- Device publishes `carrotNavi` cereal at ≥0.5 s heartbeat whenever items change → keep 1 Hz feed.
- Staleness: if the phone goes quiet, `sm.alive['carrotNavi']` lapses → carrot falls back to
  local Valhalla/navd logic automatically (fail-open).

## Item value schemas (fields the device actually reads)
- `vehicle`: `{lat, lon, heading_deg(0-360), speed_kph, road_name, virtual_gps:true}`
- `guidance_current` / `guidance_next`:
  `{distance_m, time_sec, turn_type, road_name, main_text, near_direction, mid_direction,
    far_direction, point:{lat,lon}}` — `turn_type` codes below; `guidance_current` drives the
  TBT banner, countdown, `xDistToTurn`/`xTurnInfo` (AutoTurn + cluster turn arrow).
- `speed`:
  `{current_kph, road_limit_kph,                                  // road_limit MUST be >0, ≤200, multiple of 10
    sdi:{type, distance_m, speed_limit_kph, section_type,
         block_type, block_speed_kph, block_distance_m},
    sdi_secondary:{...same...},
    section:{active, speed_limit_kph, average_kph, remaining_distance_m,
             remaining_time_sec, progress(0-1), suspended, off_route}}`
  - `sdi.type ∈ [0,1,2,3,4,7,8,75,76]` + `speed_limit_kph>0` ⇒ active limit control (xSpdLimit).
  - `sdi.block_type ∈ [2,3]` ⇒ decel point uses `block_distance_m`, target `block_speed_kph`.
  - **The next speed-limit change is `sdi`** (distance_m + new limit); the one after → `sdi_secondary`.
  - **A roundabout/turn entry target** can ride in the same `sdi` as block fields while `type=3`.
- `route`: `{remain_distance_m, remain_time_sec, moved_distance_m, moved_time_sec,
  total_distance_m, polyline:[{lat,lon}× ≤256]}`
- `navigation_status`: `{mode, guidance_active, off_route, route_present}`
- `lane_current` / `lane_ahead(array)`: `{count, distance_m, visible, current_lane, turn_code,
  turn_info[], available[], ...}` (TMAP lane arrow vocabulary — v1 sends `present:false`).
- `traffic_signal`, `crossroad`, `app_status`, `camera_state`, `composition_state`: send
  `present:false` unless you have data (traffic_signal schema exists if you want light countdowns).

## turn_type code table (device `nav_type_mapping`, = OSRM-style)
```
12 turn left        13 turn right        16 sharp left      19 sharp right
1000 slight left   1001 slight right     14 uturn          201 arrive
  6 fork right      7 fork left (also 17,44,75,76,118) / 43,73,74,117,123,124 fork right
1002 fork slight left 1003 fork slight right
101/104/111/114 off ramp slight right   102/105/112/115 off ramp slight left
107 off ramp right  106 off ramp left
131/132 rotary exit slight right        133 rotary exit right
134/135 rotary sharp right              139 rotary left
140/141 rotary slight left              136/137/138 rotary sharp left
142 rotary straight (opposite side)
```
nEventType mapping: turn=1/2, fork=3/4, **rotary=5** (cluster shows the roundabout icon + speed),
arrive/notification = display only.

## TomTom → code mapping (see tomtom2carrot.py for the machine version)
- `TURN_LEFT/RIGHT, SLIGHT_*, SHARP_*` → 12/13/1000/1001/16/19
- `MAKE_UTURN` → 14, `ARRIVE` → 201, `DEPART/WAYPOINT_*` → absent guidance
- `ROUNDABOUT_SLIGHT_LEFT/LEFT/SHARP_LEFT/STRAIGHT/SHARP_RIGHT/RIGHT/SLIGHT_RIGHT/BACK`
  → 140 / 139 / 136 / 142 / 134 / 133 / 131 / 134
- `EXIT_ROUNDABOUT` → keep the chosen exit code as `guidance_current` until ~10 m past point,
  then send `guidance_current` = next maneuver.
- `KEEP_LEFT/KEEP_RIGHT/fork-ish road names` → 1002/1003 or 6/7 (heuristic; refine per signage).
- Speed-limit sections: current limit → `road_limit_kph` (round DOWN to 10-step);
  next section change → `sdi {type:3, distance_m: meters to change, speed_limit_kph:new}`.
- Roundabout/junction entry speed (no TomTom "recommended speed" on all maneuvers):
  `v = sqrt(a_lat * R)` with a_lat ≈ 2.8 m/s² and R from the SDK curve/geometry helper;
  fallbacks: rotary 25-30, turn 30, sharp 20, off-ramp 40.

## Device settings to enable (carrot)
- `AutoNaviSpeedCtrlMode ≥ 1` (SDI speed control), `AutoNaviSpeedSafetyFactor` (105 = 5 % above limit)
- `AutoNaviSpeedBumpTime` / `AutoNaviSpeedBumpSpeed` for pre-decel shape
- `AutoTurnControl` if you want the turn-guided behavior at junctions; lane-change-at-exit
  still respects normal lane-change rules (desire system).

## Test
`feed_sim.py` simulates a scripted drive (limit change → roundabout right → junction right)
either printing frames (`--dry`) or pushing live to the comma (`--host 192.168.0.22`).
Watch the device: `python3 -c "from openpilot.cereal import messaging; sm=messaging.SubMaster(['carrotNavi']);
[ (sm.update(), print(sm['carrotNavi'].speed.sdiType, sm['carrotNavi'].speed.sdiDistanceM, sm['carrotNavi'].guidanceCurrent.turnType)) for _ in range(20)]"`
