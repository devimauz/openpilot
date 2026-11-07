// Kotlin port of tomtom2carrot.py + feed envelope rules. Kept in lockstep with the
// Python selftest (python feed_sim.py --selftest). No Android deps in this file.
package hu.tunelab.navbridge

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

object Protocol {
  const val PROTOCOL_VERSION = 2
  const val CATALOG_REVISION = 1
  const val SCHEMA_VERSION = 1

  val JSON_NAMES = listOf(
    "vehicle", "guidance_current", "guidance_next", "lane_current", "lane_ahead",
    "speed", "traffic_signal", "crossroad", "route", "navigation_status",
    "app_status", "camera_state", "composition_state",
  )
  val IMAGE_NAMES = listOf(
    "tbt_current_compact", "tbt_current_full", "tbt_next", "traffic_signal",
    "lane_top", "lane_bottom", "safety_primary", "safety_secondary", "safety_section",
    "crossroad_minimized", "crossroad_expanded", "center_tbt_icon", "center_tbt_text",
    "center_tbt_fee",
  )
  val RENDER_NAMES = listOf("map_main")
  val CATALOG: List<Pair<String, String>> =
    JSON_NAMES.map { "json" to it } + IMAGE_NAMES.map { "image" to it } + RENDER_NAMES.map { "render" to it }

  const val ROAD_LIMIT_STEP = 10
  const val MAX_ROUTE_POINTS = 256
  const val SDI_LIMIT_TYPE = 3
  const val BLOCK_CURVE_TYPE = 2
  val SPEED_CTRL_TYPES = setOf(0, 1, 2, 3, 4, 7, 8, 75, 76)
  val NO_MANEUVER = setOf("DEPART", "WAYPOINT_LEFT", "WAYPOINT_RIGHT", "WAYPOINT_REACHED", "WAYPOINT_AHEAD")

  val TURN = mapOf(
    "TURN_LEFT" to 12, "TURN_RIGHT" to 13, "SHARP_LEFT" to 16, "SHARP_RIGHT" to 19,
    "SLIGHT_LEFT" to 1000, "SLIGHT_RIGHT" to 1001, "MAKE_UTURN" to 14, "ARRIVE" to 201,
    "KEEP_LEFT" to 44, "KEEP_RIGHT" to 43,
    "ROUNDABOUT_STRAIGHT" to 142, "ROUNDABOUT_SLIGHT_RIGHT" to 131, "ROUNDABOUT_RIGHT" to 133,
    "ROUNDABOUT_SHARP_RIGHT" to 134, "ROUNDABOUT_SLIGHT_LEFT" to 140, "ROUNDABOUT_LEFT" to 139,
    "ROUNDABOUT_SHARP_LEFT" to 136, "ROUNDABOUT_BACK" to 135,
  )
  val ENTRY_SPEED = mapOf(
    12 to 30, 13 to 30, 16 to 20, 19 to 20, 1000 to 35, 1001 to 35,
    6 to 40, 7 to 40, 1002 to 40, 1003 to 40, 131 to 28, 133 to 25, 142 to 30, 139 to 28, 140 to 30,
    101 to 40, 102 to 40, 104 to 40, 105 to 40, 106 to 40, 107 to 40,
  )

  fun snapRoadLimit(kph: Int): Int {
    val v = kph / ROAD_LIMIT_STEP * ROAD_LIMIT_STEP
    return if (v in 1..200) v else 0
  }

  fun roundaboutExitCode(exitNumber: Int, numExits: Int): Int {
    if (numExits <= 1) return 142
    val frac = (exitNumber - 1).toDouble() / max(1, numExits - 1)
    return when {
      frac <= 0.15 -> 131
      frac <= 0.40 -> 133
      frac <= 0.55 -> 142
      frac <= 0.80 -> 139
      else -> 136
    }
  }

  /** TomTom turnAngleInDecimalDegrees: negative = right turn through the circle. */
  fun rotaryAngleCode(angleDeg: Double): Int {
    val a = abs(angleDeg)
    return when {
      angleDeg >= -10 && angleDeg <= 10 -> 142
      angleDeg < 0 && a <= 25 -> 131
      angleDeg < 0 && a <= 95 -> 133
      angleDeg < 0 && a <= 135 -> 134
      angleDeg < 0 -> 135
      angleDeg > 0 && a <= 25 -> 140
      angleDeg > 0 && a <= 95 -> 139
      else -> 136
    }
  }

  /** maneuver names accepted from BOTH TomTom SDK (SCREAMING) and REST guidance (camelCase). */
  fun normManeuver(raw: String): String {
    val rest = mapOf(
      "turnLeft" to "TURN_LEFT", "turnRight" to "TURN_RIGHT",
      "slightLeft" to "SLIGHT_LEFT", "slightRight" to "SLIGHT_RIGHT",
      "sharpLeft" to "SHARP_LEFT", "sharpRight" to "SHARP_RIGHT",
      "uturn" to "MAKE_UTURN", "arrive" to "ARRIVE", "depart" to "DEPART",
      "keepLeft" to "KEEP_LEFT", "keepRight" to "KEEP_RIGHT",
      "takeRoundabout" to "ROUNDABOUT_ENTRY", "exitRoundabout" to "EXIT_ROUNDABOUT",
      "roundaboutStraight" to "ROUNDABOUT_STRAIGHT", "roundaboutRight" to "ROUNDABOUT_RIGHT",
      "roundaboutSlightRight" to "ROUNDABOUT_SLIGHT_RIGHT", "roundaboutSharpRight" to "ROUNDABOUT_SHARP_RIGHT",
      "roundaboutLeft" to "ROUNDABOUT_LEFT", "roundaboutSlightLeft" to "ROUNDABOUT_SLIGHT_LEFT",
      "roundaboutSharpLeft" to "ROUNDABOUT_SHARP_LEFT", "roundaboutBack" to "ROUNDABOUT_BACK",
      "takeForkLeft" to "FORK_LEFT", "takeForkRight" to "FORK_RIGHT",
      "takeMotorwayLeft" to "OFF_RAMP_LEFT", "takeMotorwayRight" to "OFF_RAMP_RIGHT",
    )
    return rest[raw] ?: raw.uppercase()
  }
}

data class Maneuver(
  val type: String,
  var distanceM: Int = 0,
  val timeSec: Int = 0,
  val roadName: String = "",
  val mainText: String = "",
  val point: Pair<Double, Double>? = null,
  val exitNumber: Int? = null,
  val numExits: Int? = null,
  val radiusM: Double? = null,
  val junction: String? = null,
  val turnAngleDeg: Double? = null,
) {
  fun turnCode(): Int? {
    if (junction == "ROUNDABOUT") {
      turnAngleDeg?.let { return Protocol.rotaryAngleCode(it) }
      if (exitNumber != null) return Protocol.roundaboutExitCode(exitNumber, numExits ?: 4)
      return 142
    }
    val norm = Protocol.normManeuver(type)
    Protocol.TURN[norm]?.let { return it }
    if (norm == "EXIT_ROUNDABOUT" && exitNumber != null)
      return Protocol.roundaboutExitCode(exitNumber, numExits ?: 4)
    return null
  }

  fun entrySpeedKph(): Int {
    radiusM?.let { r ->
      val v = sqrt(2.8 * r) * 3.6
      return max(15, min(60, v.toInt()))
    }
    return Protocol.ENTRY_SPEED[turnCode() ?: 0] ?: 0
  }
}

data class LimitChange(val distanceM: Int, val limitKph: Int)

data class NaviSnapshot(
  var lat: Double, var lon: Double, var headingDeg: Double, var speedKph: Double,
  var currentLimitKph: Int,
  var nextChange: LimitChange? = null,
  var secondChange: LimitChange? = null,
  var maneuverNow: Maneuver? = null,
  var maneuverNext: Maneuver? = null,
  var routePolyline: List<Pair<Double, Double>> = emptyList(),
  var routeRemainM: Int = 0, var routeRemainSec: Int = 0,
  var routeMovedM: Int = 0, var routeTotalM: Int = 0,
  var guidanceActive: Boolean = true,
  var offRoute: Boolean = false,
  var roadName: String = "",
) {
  private fun guidance(m: Maneuver?): Map<String, Any?> {
    if (m == null || Protocol.NO_MANEUVER.contains(Protocol.normManeuver(m.type)))
      return mapOf("present" to false, "reason" to "source_absent")
    val code = m.turnCode() ?: return mapOf("present" to false, "reason" to "invalid")
    val value = mutableMapOf<String, Any?>(
      "distance_m" to max(0, m.distanceM), "time_sec" to max(0, m.timeSec),
      "turn_type" to code, "road_name" to m.roadName.take(96),
      "main_text" to m.mainText.take(160),
      "near_direction" to "", "mid_direction" to "", "far_direction" to "",
    )
    m.point?.let { value["point"] = mapOf("lat" to it.first, "lon" to it.second) }
    return mapOf("present" to true, "value" to value)
  }

  fun buildItems(): Map<String, Map<String, Any?>> {
    val speed = mutableMapOf<String, Any?>("current_kph" to speedKph)
    Protocol.snapRoadLimit(currentLimitKph).let { if (it > 0) speed["road_limit_kph"] = it }

    val limitCand = nextChange?.takeIf { it.limitKph > 0 }?.let { nc ->
      mapOf("type" to Protocol.SDI_LIMIT_TYPE, "distance_m" to nc.distanceM,
        "speed_limit_kph" to nc.limitKph, "section_type" to -1,
        "block_type" to Protocol.BLOCK_CURVE_TYPE,
        "block_speed_kph" to (Protocol.snapRoadLimit(nc.limitKph).let { if (it > 0) it else nc.limitKph }),
        "block_distance_m" to nc.distanceM)
    }
    val m = maneuverNow
    val mNorm = m?.let { Protocol.normManeuver(it.type) } ?: ""
    val manCand = if (m != null && mNorm.startsWith("ROUNDABOUT") && !mNorm.startsWith("ROUNDABOUT_EXIT")
        || mNorm.startsWith("EXIT_ROUNDABOUT") || mNorm.startsWith("TURN_") || mNorm.startsWith("SHARP_")
        || mNorm.startsWith("SLIGHT_") || mNorm.startsWith("OFF_RAMP") || mNorm.startsWith("FORK")
        || mNorm == "ROUNDABOUT_ENTRY") {
      val entry = m!!.entrySpeedKph()
      if (entry > 0 && m.distanceM <= 350) mapOf(
        "type" to 3, "distance_m" to m.distanceM, "speed_limit_kph" to entry,
        "section_type" to -1, "block_type" to Protocol.BLOCK_CURVE_TYPE,
        "block_speed_kph" to entry, "block_distance_m" to m.distanceM)
      else null
    } else null

    val demoted: Map<String, Any?>? = when {
      manCand != null -> { speed["sdi"] = manCand; limitCand }
      limitCand != null -> { speed["sdi"] = limitCand; null }
      else -> null
    }
    secondChange?.let { sc ->
      val demotedDist = (demoted?.get("distance_m") as? Int) ?: Int.MAX_VALUE
      if (demoted != null && demotedDist <= sc.distanceM) speed["sdi_secondary"] = demoted
      else speed["sdi_secondary"] = mapOf("type" to Protocol.SDI_LIMIT_TYPE, "distance_m" to sc.distanceM,
        "speed_limit_kph" to sc.limitKph, "section_type" to -1,
        "block_type" to -1, "block_speed_kph" to 0, "block_distance_m" to 0)
    } ?: demoted?.let { speed["sdi_secondary"] = it }

    val items = mutableMapOf<String, Map<String, Any?>>()
    items["vehicle"] = mapOf("present" to true, "value" to mapOf(
      "lat" to lat, "lon" to lon, "heading_deg" to headingDeg, "speed_kph" to speedKph,
      "road_name" to roadName.take(96), "virtual_gps" to true))
    items["guidance_current"] = guidance(maneuverNow)
    items["guidance_next"] = guidance(maneuverNext)
    items["speed"] = mapOf("present" to true, "value" to speed)
    items["route"] = mapOf("present" to true, "value" to mapOf(
      "remain_distance_m" to routeRemainM, "remain_time_sec" to routeRemainSec,
      "moved_distance_m" to routeMovedM, "total_distance_m" to routeTotalM,
      "polyline" to routePolyline.take(Protocol.MAX_ROUTE_POINTS)
        .map { mapOf("lat" to it.first, "lon" to it.second) }))
    items["navigation_status"] = mapOf("present" to true, "value" to mapOf(
      "mode" to "tomtom", "guidance_active" to guidanceActive, "off_route" to offRoute,
      "route_present" to routePolyline.isNotEmpty()))
    items["app_status"] = mapOf("present" to true, "value" to mapOf("foreground" to true))
    for (name in listOf("lane_current", "traffic_signal", "crossroad", "camera_state", "composition_state"))
      items[name] = mapOf("present" to false, "reason" to "source_absent")
    items["lane_ahead"] = mapOf("present" to false, "reason" to "source_absent", "value" to null)
    return items
  }
}
