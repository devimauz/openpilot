# NavBridge (Android) — TomTom → comma/carrot navigation feed

v0.5 app skeleton that turns your phone into the navigation brain for the comma device:
TomTom route guidance (roundabout exits, junction maneuvers, speed-limit sections + changes)
is streamed over the **carrot_navi v2 protocol** (ws://comma:7714) exactly like the Korean
TMAP feed. Device-side behavior: `AutoNaviSpeedCtrl` deceleration, roundabout/junction
banner + cluster icons, turn guidance for AutoTurn. See ../README.md for the protocol spec.

## Layout
```
android/
  settings.gradle.kts  build.gradle.kts  gradle.properties   <- put TomTomApiKey here
  app/src/main/java/hu/tunelab/navbridge/
    Protocol.kt            # exact port of tomtom2carrot.py (validated by python selftest)
    NavSource.kt           # source interface
    TomTomRestNavSource.kt # v0.5: TomTom Routing REST (guidance + SpeedLimit sections) + phone GPS
    FeedClient.kt          # control-ws handshake + 13 item websockets, seq-managed
    BridgeService.kt       # foreground service wiring source -> feed
    MainActivity.kt        # IP + destination form, start/stop, status
  app/src/test/.../ProtocolSelfTest.kt   # mirrors python feed_sim --selftest
```

## Build & run (first time)
1. Install Android Studio, open the `android/` folder, let it create the Gradle wrapper
   (File → Sync). Nothing else needed for v0.5 — no proprietary SDK artifacts.
2. Grab a free key at developer.tomtom.com → put it in `gradle.properties`
   (`TomTomApiKey=...`).
3. Run `ProtocolSelfTest` (should pass — same asserts as the Python twin) and
   `python ../feed_sim.py --selftest` for parity.
4. Install on the phone.

## Drive setup
- **Home / test:** comma on home Wi-Fi → enter its LAN IP (e.g. 192.168.0.22).
- **On the road:** the comma must be reachable by the phone. Simplest: phone Wi-Fi hotspot
  ON, comma joins it (comma settings → network), comma gets a 192.168.43.x IP — enter that
  (shown in carrot web settings → network). Carrot Vision's discovery UDP :7705 broadcast
  can also be used later to auto-pick the IP.
- Destination format: `lat,lon` (paste from TomTom app share or any map). Route is fetched
  once at START from the phone's current GPS; re-press START to reroute.

## What to watch on the comma
- carrot web → debug/carrotNavi: `sdiType/sdiDistanceM/sdiSpeedLimitKph` count down,
  `naviEventType=5` (rotary) / `1-2` (turn) near maneuvers, cluster shows the roundabout
  icon and target speed; `xSpdLimit` drops toward the entry speed ahead of the roundabout.
- If the phone feed stops, device falls back to local navd/Valhalla automatically.

## Device params to set (carrot settings)
`AutoNaviSpeedCtrlMode ≥ 1`, `AutoNaviSpeedBumpTime`, `AutoNaviSpeedSafetyFactor` (~105),
`AutoTurnControl` if you want guided turns; verify `EnableRadarTracks` etc. are untouched.

## v1.1 — real TomTom SDK path (better than REST matching)
The REST source computes progress by projecting GPS onto the polyline — fine, but the TomTom
**Maps SDK for Android** gives you SDK-matched progress, live `RoundaboutInstruction.exitNumber`,
lane guidance and re-routing callbacks. To add it:
1. settings.gradle.kts: uncomment the TomTom maven repo; app deps: uncomment `android-sdk`.
2. Create `TomTomSdkNavSource.kt` implementing `NavSource`, mirroring TomTom's docs
   (RoutePlanService.calculateRoute + GuidanceEngine listeners); its callbacks map 1:1 into
   `NaviSnapshot.maneuverNow/next` (their maneuver names already match `Protocol.TURN` keys
   like `ROUNDABOUT_RIGHT`), and `route.sections.speedLimitSections` fill
   `currentLimitKph/nextChange/secondChange`. Keep FeedClient/BridgeService unchanged.

## Known v0.5 edges to verify on the first real run
- TomTom REST guidance field names (`routePoints`, `sections.speedLimit`, `maneuver.exitNumber`)
  are parsed defensively; if a section comes back empty, dump one `calculateRoute` response
  (browser) and adjust `parseRoute()` — the mapping logic itself is protocol-tested.
- `roughRadius()` is a heuristic for entry speed; TomTom SDK gives proper curve data later.
- Roundabout *through* guidance: while inside the circle the SDK keeps the exit as the active
  instruction; REST emulation keeps `EXIT_ROUNDABOUT` until its point is passed — matches.
