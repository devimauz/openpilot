// TomTom Routing API (guidance + speedLimit sections) -> NaviSnapshot at ~1 Hz.
// No proprietary SDK needed; same data the SDK exposes for route guidance.
package hu.tunelab.navbridge

import android.content.Context
import android.location.Location
import com.google.android.gms.location.LocationCallback
import com.google.android.gms.location.LocationRequest
import com.google.android.gms.location.LocationResult
import com.google.android.gms.location.Priority
import com.google.gson.Gson
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import java.net.HttpURLConnection
import java.net.URL
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

class TomTomRestNavSource(
  private val context: Context,
  private val apiKey: String,
  private val onUpdate: (NaviSnapshot) -> Unit,
  private val onLog: (String) -> Unit,
) : NavSource {
  data class RoutePt(val lat: Double, val lon: Double)
  private data class Instruction(val type: String, val pointIndex: Int, val distanceAlongM: Double,
                                 val street: String, val exitNumber: Int?, val text: String)

  private var routePoints: List<RoutePt> = emptyList()
  private var cum: DoubleArray = DoubleArray(0)                 // meters at each route point
  private var totalM = 0
  private var instructions: List<Instruction> = emptyList()
  private var limitSections: List<Triple<Int, Int, Int>> = emptyList()  // startIdx, endIdx, kph
  private var lastLoc: Location? = null
  private var movedM = 0.0
  private val fused = com.google.android.gms.location.LocationServices
    .getFusedLocationProviderClient(context)
  private val scope = CoroutineScope(Dispatchers.Default)
  private var loopJob: Job? = null

  override fun startRoute(destLat: Double, destLon: Double) {
    scope.launch(Dispatchers.IO) {
      val origin = lastLoc
      if (origin == null) { onLog("no fix yet — wait 5 s and retry"); return@launch }
      val url = "https://api.tomtom.com/routing/1/calculateRoute/" +
        "${origin.latitude},${origin.longitude}:$destLat,$destLon/json" +
        "?guidance=true&sectionTypes=SpeedLimit&maxPoints=512&travelMode=thoroughFair"
      try {
        val body = URL(url).let { (it.openConnection() as HttpURLConnection).apply {
          connectTimeout = 8000; readTimeout = 10000 } }
          .inputStream.bufferedReader().readText()
        parseRoute(body)
        onLog("route ok: ${routePoints.size} pts, ${instructions.size} maneuvers, " +
          "${limitSections.size} limit sections, ${totalM} m")
      } catch (e: Exception) {
        onLog("route failed: ${e.message}")
      }
    }
    startLocation()
    loopJob?.cancel()
    loopJob = scope.launch { while (true) { tick(); delay(1000) } }
  }

  override fun stop() { loopJob?.cancel(); fused.removeLocationUpdates(cb) }

  private val cb = object : LocationCallback() {
    override fun onLocationResult(r: LocationResult) { r.lastLocation?.let { lastLoc = it } }
  }
  private fun startLocation() {
    fused.requestLocationUpdates(
      LocationRequest.Builder(Priority.PRIORITY_HIGH_ACCURACY, 1000L).build(), cb, context.mainLooper)
  }

  @Suppress("UNCHECKED_CAST")
  private fun parseRoute(json: String) {
    val root = Gson().fromJson(json, Map::class.java) as Map<String, Any?>
    val routes = (root["routes"] as? List<Map<String, Any?>>).orEmpty()
    val route = routes.firstOrNull() ?: error("no route")
    val summary = route["summary"] as Map<String, Any?>
    totalM = ((summary["lengthInMeters"] as? Double)?.toInt()) ?: 0
    val geom = (route["routePoints"] as List<Map<String, Any?>>).map {
      RoutePt((it["latitude"] as Double), (it["longitude"] as Double))
    }
    routePoints = geom
    cum = DoubleArray(geom.size)
    for (i in 1 until geom.size) cum[i] = cum[i - 1] + meters(geom[i - 1], geom[i])

    instructions = buildList {
      val guidance = (route["guidance"] as? Map<String, Any?>)
        ?.let { it["instructions"] as? List<Map<String, Any?>> }.orEmpty()
      for (ins in guidance) {
        val maneuver = ins["maneuver"] as? Map<String, Any?> ?: continue
        val startPoint = (ins["startPointIndex"] as? Double)?.toInt() ?: continue
        val type = maneuver["type"]?.toString() ?: continue
        val exit = (maneuver["exitNumber"] as? Double)?.toInt()
        val street = (ins["street"] as? List<Map<String, Any?>>)?.firstOrNull()?.get("value")?.toString() ?: ""
        add(Instruction(type, startPoint, cum.getOrElse(startPoint) { 0.0 }, street, exit,
          ins["instructionDetails"]?.toString() ?: ""))
      }
    }
    limitSections = buildList {
      val sections = ((route["sections"] as? Map<String, Any?>)?.get("speedLimit") as? List<Map<String, Any?>>).orEmpty()
      for (s in sections) {
        val start = (s["startPointIndex"] as? Double)?.toInt() ?: continue
        val end = (s["endPointIndex"] as? Double)?.toInt() ?: continue
        val max = ((s["maxSpeedLimit"] as? Map<String, Any?>)?.get("toValue") as? Double)?.toInt() ?: continue
        add(Triple(start, end, max))
      }
    }
  }

  private fun tick() {
    val loc = lastLoc ?: return
    if (routePoints.isEmpty()) return
    movedM = projectProgress(loc.latitude, loc.longitude)
    val remaining = (totalM - movedM).coerceAtLeast(0.0)

    // limit at position + next change
    val here = limitSections.firstOrNull { movedM >= cum[it.first] && movedM < cum[it.second] }
    val upcoming = limitSections
      .filter { cum[it.first] > movedM }
      .sortedBy { cum[it.first] }
    val nextChange = upcoming.firstOrNull()?.let { LimitChange((cum[it.first] - movedM).toInt(), it.third) }
    val secondChange = upcoming.drop(1).firstOrNull()?.let { LimitChange((cum[it.first] - movedM).toInt(), it.third) }

    val ahead = instructions
      .filter { it.distanceAlongM > movedM - 15 && Protocol.normManeuver(it.type) != "DEPART" }
      .sortedBy { it.distanceAlongM }
    val now = ahead.firstOrNull()?.let { m ->
      Maneuver(m.type, distanceM = (m.distanceAlongM - movedM).toInt(),
        roadName = m.street, exitNumber = m.exitNumber, mainText = m.text,
        point = routePoints.getOrNull(m.pointIndex)?.let { it.lat to it.lon },
        radiusM = roughRadius(m))
    }
    val nxt = ahead.drop(1).firstOrNull()?.let { m ->
      Maneuver(m.type, distanceM = (m.distanceAlongM - movedM).toInt(), roadName = m.street,
        exitNumber = m.exitNumber, point = routePoints.getOrNull(m.pointIndex)?.let { it.lat to it.lon })
    }

    val snap = NaviSnapshot(
      lat = loc.latitude, lon = loc.longitude,
      headingDeg = if (loc.hasBearing()) loc.bearing.toDouble() else 0.0,
      speedKph = if (loc.hasSpeed()) loc.speed * 3.6 else 0.0,
      currentLimitKph = here?.third ?: 0,
      nextChange = nextChange, secondChange = secondChange,
      maneuverNow = now, maneuverNext = nxt,
      routePolyline = routePoints.map { it.lat to it.lon },
      routeRemainM = remaining.toInt(), routeMovedM = movedM.toInt(), routeTotalM = totalM,
      guidanceActive = true, offRoute = offRoute(loc.latitude, loc.longitude),
      roadName = (now?.roadName ?: "").ifEmpty { here?.let { "" } ?: "" },
    )
    onUpdate(snap)
  }

  private fun roughRadius(m: Instruction): Double? {
    // estimate turn radius from the polyline: sample speed of heading change around maneuver point
    val idx = m.pointIndex
    if (idx <= 3 || idx + 3 >= routePoints.size) return null
    var headingSum = 0.0
    for (i in idx - 2 until idx + 2) {
      val a = bearing(routePoints[i], routePoints[i + 1])
      val b = bearing(routePoints[i + 1], routePoints[i + 2])
      headingSum += angleDiff(a, b)
    }
    val turn = Math.toRadians(kotlin.math.abs(headingSum))
    if (turn < 0.2) return null
    val arc = (cum[minOf(idx + 2, cum.size - 1)] - cum[maxOf(idx - 2, 0)]).coerceAtLeast(1.0)
    return arc / turn
  }

  private fun projectProgress(lat: Double, lon: Double): Double {
    var best = 0.0; var bestD = Double.MAX_VALUE
    for (i in 0 until routePoints.size - 1) {
      val d = pointSegMeters(lat, lon, routePoints[i], routePoints[i + 1])
      if (d < bestD) { bestD = d; best = cum[i] + meters(routePoints[i], routePoints[i + 1]) * 0.5 }
    }
    return best
  }

  private fun offRoute(lat: Double, lon: Double): Boolean {
    var best = Double.MAX_VALUE
    for (i in 0 until routePoints.size - 1)
      best = minOf(best, pointSegMeters(lat, lon, routePoints[i], routePoints[i + 1]))
    return best > 50
  }

  private fun meters(a: RoutePt, b: RoutePt) = hypot(
    Math.toRadians(b.lat - a.lat) * 6371000.0,
    Math.toRadians(b.lon - a.lon) * 6371000.0 * cos(Math.toRadians((a.lat + b.lat) / 2)))

  private fun bearing(a: RoutePt, b: RoutePt): Double {
    val y = sin(Math.toRadians(b.lon - a.lon)) * cos(Math.toRadians(b.lat))
    val x = cos(Math.toRadians(a.lat)) * sin(Math.toRadians(b.lat)) -
      sin(Math.toRadians(a.lat)) * cos(Math.toRadians(b.lat)) * cos(Math.toRadians(b.lon - a.lon))
    return Math.toDegrees(atan2(y, x)).let { (it + 360) % 360 }
  }

  private fun angleDiff(a: Double, b: Double): Double {
    var d = b - a
    while (d > 180) d -= 360
    while (d < -180) d += 360
    return d
  }

  private fun pointSegMeters(lat: Double, lon: Double, a: RoutePt, b: RoutePt): Double {
    // flat-earth approximation fine at segment scale
    val kx = 111320.0 * cos(Math.toRadians(lat))
    val px = (lon - a.lon) * kx
    val py = (lat - a.lat) * 110540.0
    val dx = (b.lon - a.lon) * kx
    val dy = (b.lat - a.lat) * 110540.0
    val len2 = dx * dx + dy * dy
    val t = if (len2 == 0.0) 0.0 else ((px * dx + py * dy) / len2).coerceIn(0.0, 1.0)
    return hypot(px - t * dx, py - t * dy)
  }
}
